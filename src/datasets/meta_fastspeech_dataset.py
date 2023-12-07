import os
from collections import defaultdict
from tqdm import tqdm

import random
import torch

from src.datasets.fastspeech_dataset import FastSpeechDataset


class MetaFastSpeechDataset(FastSpeechDataset):
    def __init__(
        self,
        path_to_transcript_dict,
        acoustic_checkpoint_path,
        cache_dir,
        lang,
        n_way,
        k_shot,
        k_query,
        n_task,
        n_batch=1,
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
        super(MetaFastSpeechDataset, self).__init__(
            path_to_transcript_dict,
            acoustic_checkpoint_path,
            cache_dir,
            lang,
            loading_processes,
            min_len_in_seconds,
            max_len_in_seconds,
            cut_silence,
            reduction_factor,
            device,
            rebuild_cache,
            ctc_selection,
            save_imgs,
        )

        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_task = n_task
        self.n_batch = n_batch

        #  categorize speaker
        self.speaker_samples = defaultdict(list)
        for datapoint in tqdm(self.datapoints):
            speaker_id = os.path.basename(datapoint[8]).split("-")[0]
            self.speaker_samples[speaker_id].append((
            datapoint[0],
            datapoint[1],
            datapoint[2],
            datapoint[3],
            datapoint[4],
            datapoint[5],
            datapoint[6],
            datapoint[7],
            self.language_id,
            speaker_id,
            datapoint[8]
        ))

        # remove speaker with few samples
        discard_speaker = []
        for speaker_id, samples in self.speaker_samples.items():
            if len(samples) < max(self.k_shot, self.k_query):
                print(f"Remove speaker {speaker_id} with {len(samples)} samples")
                discard_speaker.append(speaker_id)
        for speaker_id in discard_speaker:
            self.speaker_samples.pop(speaker_id, None)

        if 2 * self.n_way > len(self.speaker_samples):
            raise Exception(
                f"Number of speaker must not less than {2*self.n_way}, but got {len(self.speaker_samples)}"
            )
        else:
            print(f"Got {len(self.speaker_samples)} speakers")

    def __getitem__(self, index):
        # support set's size : (self.n_task, self.k_shot  * self.n_way)
        # query set's size   : (self.n_task, self.k_query * self.n_way)
        spts, qrys = [], []
        for _ in range(self.n_task):
            spt, qry = [], []
            selected_spk = random.sample(list(self.speaker_samples.keys()), 2 * self.n_way)
            for i, speaker_id in enumerate(selected_spk[: self.n_way]):
                selected_samples = random.sample(
                    self.speaker_samples[speaker_id], self.k_shot
                )
                spt += list(selected_samples)

            for i, speaker_id in enumerate(selected_spk[self.n_way :]):
                selected_samples = random.sample(
                    self.speaker_samples[speaker_id], self.k_query
                )
                qry += list(selected_samples)

            random.shuffle(spt)
            random.shuffle(qry)
            
            spts.append(spt)
            qrys.append(qry)

        return spts, qrys
    
    def next(self):
        return self[0]

    def __len__(self):
        return self.n_batch


if __name__ == "__main__":
    dataset = MetaFastSpeechDataset(
        path_to_transcript_dict=transcript_dict,
        acoustic_checkpoint_path=ALIGNER_CHECKPOINT,  # path to aligner.pt
        cache_dir="./librispeech/",
        lang="en",
        loading_processes=2,  # depended on how many CPU you have
        device=device,
        n_way=1,
        k_shot=3,
        k_query=1,
        n_batch=5,
    )
