import argparse

import torch

from src.datasets.fastspeech_dataset import (
    build_path_to_transcript_dict_libri_tts,
    FastSpeechDataset,
)


def build_cache(config):
    # Enable GPU if available
    if config.enable_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("GPU is not available, using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    transcript_dict = build_path_to_transcript_dict_libri_tts(config.data_dir)

    FastSpeechDataset(
        path_to_transcript_dict=transcript_dict,
        acoustic_checkpoint_path=config.alirgner_checkpoint,
        cache_dir=config.cache_dir,
        lang="en",
        loading_processes=config.n_processes,
        device=device,
        rebuild_cache=config.rebuild_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        help="Path to the dataset directory, ex: ./data/librispeech/test-clean",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--alirgner_checkpoint",
        type=str,
        help="Path to the aligner checkpoint",
        default="./weights/aligner.pt",
    )
    parser.add_argument(
        "-c",
        "--cache_dir",
        type=str,
        help="Path to the cache directory",
        default=".cache/",
    )
    parser.add_argument(
        "-g",
        "--enable_gpu",
        action="store_true",
        help="Enable GPU if available",
        default=False,
    )
    parser.add_argument(
        "-r",
        "--rebuild_cache",
        action="store_true",
        help="Rebuild the cache",
        default=False,
    )
    parser.add_argument(
        "-p",
        "--n_processes",
        type=int,
        help="Number of processes to load the data, depended on how many CPU you have",
        default=1,
    )

    config = parser.parse_args()
    print("Configurations:")
    for key, value in vars(config).items():
        print(f"  --{key}: {value}")

    build_cache(config)
