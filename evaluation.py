import os

from glob import glob
import csv
import argparse

import torch
import numpy as np
import nemo.collections.asr as nemo_asr
import torchaudio
from tqdm import tqdm

from constant import METRIC_OPTIONS, MODEL_OPTIONS, ASR_CHECKPOINT
from inference import _inference
from src.preprocessing.audio_processing import AudioPreprocessor
from src.spk_embedding.StyleEmbedding import StyleEmbedding
from src.tts.vocoders.hifigan.HiFiGAN import HiFiGANGenerator

result_dir = "eval_results"
os.makedirs(result_dir, exist_ok=True)

@torch.no_grad()
def wer_evaluate(path_to_syn_folder, eval_sentences):
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=ASR_CHECKPOINT["wer"])
    wer = list()
    for spk_id in os.listdir(path_to_syn_folder):
        file_paths = [f"{path_to_syn_folder}/{spk_id}/{i}.wav" for i in range(len(eval_sentences))]
        transcribes = asr_model.transcribe(file_paths)
        wer.append(nemo_asr.metrics.wer.word_error_rate(transcribes, eval_sentences))

    wer = np.array(wer)

    # Write to csv file
    with open(f"{result_dir}/wer_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mean", "Std"])
        writer.writerow([wer.mean(), wer.std()])

    print("Mean - ", wer.mean(), "\nStd - ", wer.std())
    
    return wer.mean(), wer.std()

@torch.no_grad()
def compute_similarity(speaker_model, X, Y):
    """
        speaker_model: a speaker label model
        X: list of paths to audio wav files of speaker 1
        Y: list of paths to audio wav files of speaker 2 
    """
    scores = []
    for x_file, y_file in zip(X,Y):
        x = speaker_model.get_embedding(x_file).squeeze()
        y = speaker_model.get_embedding(y_file).squeeze()
        # Length Normalize
        x = x / torch.linalg.norm(x)
        y = y / torch.linalg.norm(y)
        # Score
        similarity_score = torch.dot(x, y) / ((torch.dot(x, x) * torch.dot(y, y)) ** 0.5) 
        similarity_score = (similarity_score + 1) / 2
        scores.append(similarity_score.item())
    
    return scores

@torch.no_grad()
def secs_evaluate(path_to_syn_folder, path_to_speaker_folder):
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(ASR_CHECKPOINT["secs"])
    secs = list()
    for spk_id in os.listdir(path_to_speaker_folder):
        gold_file = glob(os.path.join(path_to_speaker_folder, spk_id) + "/*/*.wav")[0]
        synth_files = glob(os.path.join(path_to_syn_folder, spk_id) + "/*.wav")
        gold_files = [gold_file for _ in range(len(synth_files))]
        scores = compute_similarity(speaker_model, gold_files, synth_files)
        secs.append(np.array(scores).mean())

    # Write to csv file
    with open(f"{result_dir}/secs_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mean", "Std"])
        writer.writerow([np.array(secs).mean(), np.array(secs).std()])

    print("Mean - ", np.array(secs).mean(), "\nStd - ", np.array(secs).std())

    return np.array(secs).mean(), np.array(secs).std()

def _synthesize(
        model,
        path_to_eval_speakers: str,
        path_to_out: str,
        eval_sentences: list,
        ap,
        vocoder,
        style_embed_function,
        device="cpu",
        ):

    # Synthesize
    for spk_id in tqdm(os.listdir(path_to_eval_speakers)):
        os.makedirs(f"{path_to_out}/{spk_id}", exist_ok=True)
        ref_audio = glob(os.path.join(path_to_eval_speakers, spk_id) + "/*/*.wav")[0]
        for sent_id, sentence in enumerate(eval_sentences):
            waveform = _inference(
                text=sentence,
                ref_path=ref_audio,
                acoustic_model=model,
                ap=ap,
                vocoder=vocoder,
                style_embed_function=style_embed_function,
                device=device
                )
            torchaudio.save(
                f"{path_to_out}/{spk_id}/{sent_id}.wav",
                src=waveform,
                sample_rate=24000
                )


def evaluate(config):
    # Enable GPU if available
    if config.enable_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("GPU is not available, using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Load audio processor
    ap = AudioPreprocessor(input_sr=16000, output_sr=16000, melspec_buckets=80,
        hop_length=256,n_fft=1024,cut_silence=False,device=device)
    
    # Load checkpoints
    style_embed_function = StyleEmbedding().to(device)
    style_embed_check_dict = torch.load(config.embedding_function_checkpoint, map_location=device)
    style_embed_function.load_state_dict(style_embed_check_dict["style_emb_func"])
    style_embed_function.eval()
    style_embed_function.requires_grad_(False)

    vocoder = HiFiGANGenerator().to(device)
    avocodo_check_dict = torch.load(config.avocodo_checkpoint, map_location=device)
    vocoder.load_state_dict(avocodo_check_dict["generator"])
    vocoder.eval()
    
    # Load model
    model = MODEL_OPTIONS[config.model]["model"]().to(config.device)
    model_check_dict = torch.load(config.pretrained_checkpoint, map_location=config.device)
    model.load_state_dict(model_check_dict["model"])
    model.eval()
    model.requires_grad_(False)

    # Load eval sentences
    with open(config.path_to_eval_sentences, "r") as f:
        eval_sentences = f.read().splitlines()

    # Path to ouput synthesized audios folder
    path_to_syn_folder = f"{config.path_to_out_folder}/{config.model}/{config.metric}"

    # Synthesize audios for evaluation
    _synthesize(
        model=model,
        path_to_eval_speakers=config.path_to_eval_speakers,
        path_to_out=path_to_syn_folder,
        eval_sentences=eval_sentences,
        ap=ap,
        vocoder=vocoder,
        style_embed_function=style_embed_function,
        device=device
        )

    # Evaluate
    if config.metric == METRIC_OPTIONS["wer"]:
        return wer_evaluate(path_to_syn_folder, eval_sentences)
    elif config.metric == METRIC_OPTIONS["secs"]:
        return secs_evaluate(
            path_to_syn_folder,
            path_to_speaker_folder=config.path_to_eval_speakers)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help=f"Model to evaluate, options: {MODEL_OPTIONS.keys()}",
        choices=MODEL_OPTIONS.keys(),
    )
    parser.add_argument(
        "-me",
        "--metric",
        type=str,
        help=f"Metric to evaluate, options: {METRIC_OPTIONS.keys()}",  
    )
    parser.add_argument(
        "-pg",
        "--path_to_eval_speakers",
        type=str,
        help="Path to the eval speakers folder",
        required=True
    )
    parser.add_argument(
        "-ps",
        "--path_to_eval_sentences",
        type=str,
        help="Path to the eval sentences",
        required=True
    )
    parser.add_argument(
        "-o",
        "--path_to_out_folder",
        type=str,
        help="Path to the synthesized output folder",
        default="./synthesized",
    )
    parser.add_argument(
        "-mc"
        "--pretrained_checkpoint",
        type=str,
        help="Pretrained checkpoint path",
        required=True
    )
    parser.add_argument(
        "-se",   
        "--embedding_function_checkpoint",
        type=str,
        help="Style embedding checkpoint path",
        default="./weights/embedding_function.pt",
    )
    parser.add_argument(
        "-av",
        "--avocodo_checkpoint",
        type=str,
        help="Avocodo checkpoint path",
        default="./weights/Avocodo.pt",
    )
    parser.add_argument(
        "-g",
        "--enable_gpu",
        action="store_true",
        help="Enable GPU if available",
        default=False,
    )

    config = parser.parse_args()
    print("Configurations:")
    for key, value in vars(config).items():
        print(f"  --{key}: {value}")

    # Evaluate
    evaluate(config)