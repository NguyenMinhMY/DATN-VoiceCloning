import os
import statistics
from glob import glob

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import soundfile as sf
from numpy import trim_zeros

from src.utility.tokenizer import get_language_id
from src.utility.articulatory_features import get_feature_to_index_lookup
from src.utility.energy_calculator import EnergyCalculator
from src.utility.pitch_calculator import Parselmouth
from src.aligner.AlignerDataset import AlignerDataset
from src.preprocessing.audio_processing import AudioPreprocessor


def build_path_to_transcript_dict_libri_tts(
    root_path, meta_files=None, ignored_speakers=None
):
    """https://ai.google/tools/datasets/libri-tts/"""
    path_to_transcript = {}
    if not meta_files:
        meta_files = glob(f"{root_path}/**/*trans.txt", recursive=True)
    else:
        if isinstance(meta_files, str):
            meta_files = [os.path.join(root_path, meta_files)]

    for meta_file in meta_files:
        _meta_file = os.path.basename(meta_file).split(".")[0]
        with open(meta_file, "r", encoding="utf-8") as ttf:
            for line in ttf:
                file_name, *text = line.split(" ")
                text = " ".join(text)
                speaker_name, chapter_id, *_ = file_name.split("-")
                _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
                audio_file = os.path.join(_root_path, file_name + ".flac")
                # ignore speakers
                if isinstance(ignored_speakers, list):
                    if speaker_name in ignored_speakers:
                        continue
                path_to_transcript[audio_file] = text.lower()
                
    for audio_file in path_to_transcript:
        assert os.path.exists(
            audio_file
        ), f" [!] wav files don't exist - {audio_file}"

    return path_to_transcript


class FastGlowDataset(Dataset):
    def __init__(
        self,
        path_to_transcript_dict,
        cache_dir,
        lang,
        loading_processes=os.cpu_count() if os.cpu_count() is not None else 30,
        min_len_in_seconds=1,
        max_len_in_seconds=20,
        cut_silence=False,
        reduction_factor=1,
        device=torch.device("cpu"),
        rebuild_cache=False,
        ctc_selection=True,
        save_imgs=False,
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        if (
            not os.path.exists(os.path.join(cache_dir, "fast_train_cache.pt"))
            or rebuild_cache
        ):
            if (
                not os.path.exists(os.path.join(cache_dir, "aligner_train_cache.pt"))
                or rebuild_cache
            ):
                AlignerDataset(
                    path_to_transcript_dict=path_to_transcript_dict,
                    cache_dir=cache_dir,
                    lang=lang,
                    loading_processes=loading_processes,
                    min_len_in_seconds=min_len_in_seconds,
                    max_len_in_seconds=max_len_in_seconds,
                    cut_silences=cut_silence,
                    rebuild_cache=rebuild_cache,
                    device=device,
                )
            datapoints = torch.load(
                os.path.join(cache_dir, "aligner_train_cache.pt"), map_location="cpu"
            )
            # we use the aligner dataset as basis and augment it to contain the additional information we need for fastspeech.
            dataset = datapoints[0]
            norm_waves = datapoints[1]
            # index 2 are the speaker embeddings used for the reconstruction loss of the Aligner, we don't need them anymore
            filepaths = datapoints[3]

            print("... building dataset cache ...")
            self.datapoints = list()
            self.ctc_losses = list()

            # ==========================================
            # actual creation of datapoints starts here
            # ==========================================

            parsel = Parselmouth(reduction_factor=reduction_factor, fs=16000, use_token_averaged_f0=False)
            energy_calc = EnergyCalculator(reduction_factor=reduction_factor, fs=16000, use_token_averaged_energy=False)
            
            _, sr = sf.read(filepaths[0])
            ap = AudioPreprocessor(
                input_sr=sr,
                output_sr=16000,
                melspec_buckets=80,
                hop_length=256,
                n_fft=1024,
                cut_silence=cut_silence,
                device=device,
            )

            for index in tqdm(range(len(dataset))):
                # norm_wave = norm_waves[index]
                path_to_audio =  filepaths[index]
                wave, _ = sf.read(path_to_audio)

                norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
                norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))

                norm_wave_length = torch.LongTensor([len(norm_wave)])

                if len(norm_wave) / 16000 < min_len_in_seconds and ctc_selection:
                    continue

                text = dataset[index][0]
                melspec = dataset[index][2]
                melspec_length = dataset[index][3]

                # We deal with the word boundaries by having 2 versions of text: with and without word boundaries.
                # We note the index of word boundaries and insert durations of 0 afterwards
                text_without_word_boundaries = list()
                indexes_of_word_boundaries = list()
                for phoneme_index, vector in enumerate(text):
                    if vector[get_feature_to_index_lookup()["word-boundary"]] == 0:
                        text_without_word_boundaries.append(vector.numpy().tolist())
                    else:
                        indexes_of_word_boundaries.append(phoneme_index)

                cached_energy = (
                    energy_calc(
                        input_waves=norm_wave.unsqueeze(0),
                        input_waves_lengths=norm_wave_length,
                        feats_lengths=melspec_length,
                        text=text,
                    )[0]
                    .squeeze(0)
                    .cpu()
                )

                cached_pitch = (
                    parsel(
                        input_waves=norm_wave.unsqueeze(0),
                        input_waves_lengths=norm_wave_length,
                        feats_lengths=melspec_length,
                        text=text,
                    )[0]
                    .squeeze(0)
                    .cpu()
                )

                prosodic_condition = None

                self.datapoints.append(
                    [
                        dataset[index][0],
                        dataset[index][1],
                        dataset[index][2],
                        dataset[index][3],
                        cached_energy,
                        cached_pitch,
                        prosodic_condition, 
                        filepaths[index],
                    ]
                )
                # self.ctc_losses.append(ctc_loss)

            # =============================
            # done with datapoint creation
            # =============================

            # save to cache
            if len(self.datapoints) > 0:
                torch.save(
                    self.datapoints, os.path.join(cache_dir, "fast_train_cache.pt")
                )
            else:
                import sys

                print("No datapoints were prepared! Exiting...")
                sys.exit()
        else:
            # just load the datapoints from cache
            self.datapoints = torch.load(
                os.path.join(cache_dir, "fast_train_cache.pt"), map_location="cpu"
            )

        self.cache_dir = cache_dir
        self.language_id = get_language_id(lang)
        print(
            f"Prepared a FastGlow dataset with {len(self.datapoints)} datapoints in {cache_dir}."
        )

    def __getitem__(self, index):
        speaker_id = os.path.basename(self.datapoints[index][7]).split('-')[0]
        return (
            self.datapoints[index][0], # text
            self.datapoints[index][1], # text length
            self.datapoints[index][2], # melspec
            self.datapoints[index][3], # melspec length
            self.datapoints[index][4], # energy
            self.datapoints[index][5], # pitch
            self.datapoints[index][6], # prosodic condition
            self.language_id,
            speaker_id,
            self.datapoints[index][7] # filepath
        )

    def __len__(self):
        return len(self.datapoints)

    def remove_samples(self, list_of_samples_to_remove):
        for remove_id in sorted(list_of_samples_to_remove, reverse=True):
            self.datapoints.pop(remove_id)
        torch.save(self.datapoints, os.path.join(self.cache_dir, "fast_train_cache.pt"))
        print("Dataset updated!")