import datetime
import argparse

import wandb
import torch

from src.datasets.fastspeech_dataset import FastSpeechDataset
from constant import MODEL_OPTIONS


def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train(config):
    # Validate the configuration
    if config.phase_1_steps <= 0 and config.phase_2_steps <= 0:
        raise ValueError("Phase 1 steps or Phase 2 steps must be greater than 0")

    # Enable GPU if available
    if config.enable_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("GPU is not available, using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Enable wandb logging
    if config.wandb:
        wandb.init(project=f"voice-clone-{get_current_time()}")

    # Load the dataset
    dataset = FastSpeechDataset(
        path_to_transcript_dict=None,
        acoustic_checkpoint_path=None,
        cache_dir=config.cache_dir,
        lang="en",
        loading_processes=1,  # depended on how many CPU you have
        device=device,
    )

    # Load the model and train loop
    model = MODEL_OPTIONS[config.model]["model"]()
    train_loop = MODEL_OPTIONS[config.model]["train_loop"]

    # Train the model
    train_loop(
        model,
        dataset,
        lr=config.learning_rate,
        batch_size=max(config.batch_size, 1),
        warmup_steps=max(config.warmup_steps, 1),
        phase_1_steps=max(config.phase_1_steps, 0),
        phase_2_steps=max(config.phase_2_steps, 0),
        path_to_embed_model=config.embedding_function_checkpoint,
        save_directory=config.save_directory,
        path_to_checkpoint=config.pretrained_checkpoint,
        steps_per_save=max(config.steps_per_save, 1),
        use_wandb=config.wandb,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help=f"Model to train, options: {MODEL_OPTIONS.keys()}",
        choices=MODEL_OPTIONS.keys(),
    )
    parser.add_argument(
        "-c",
        "--cache_dir",
        type=str,
        help="Path to the cache directory, default: .cache/",
        default=".cache/",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="Learning rate, default: 1e-3",
        default=1e-3,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "-ws",
        "--warmup_steps",
        type=int,
        help="Number of warmup steps, default: 0",
        default=0,
    )
    parser.add_argument(
        "-s1",
        "--phase_1_steps",
        type=int,
        help="Number of steps for phase 1",
    )
    parser.add_argument(
        "-s2",
        "--phase_2_steps",
        type=int,
        help="Number of steps for phase 2",
    )

    parser.add_argument(
        "-e",
        "--embedding_function_checkpoint",
        type=str,
        help="Path to the embedding function checkpoint, default: ./weights/embedding_function.pt",
        default="./weights/embedding_function.pt",
    )
    parser.add_argument(
        "-mc",
        "--pretrained_checkpoint",
        type=str,
        help="Path to the model checkpoint, if not provided, the model will be trained from scratch",
        default=None,
    )

    parser.add_argument(
        "-sd",
        "--save_directory",
        type=str,
        help="Path to save directory, default: ./weights/checkpoint_models_{time}",
        default=f"./weights/checkpoint_models_{get_current_time()}",
    )
    parser.add_argument(
        "-sps",
        "--steps_per_save",
        type=int,
        help="Number of steps per save",
        default=1,
    )

    parser.add_argument(
        "-g",
        "--enable_gpu",
        action="store_true",
        help="Enable GPU if available",
        default=False,
    )
    parser.add_argument(
        "-w",
        "--wandb",
        action="store_true",
        help="Enable wandb logging, requires WANDB_API_KEY env variable",
        default=False,
    )

    config = parser.parse_args()
    print("Configurations:")
    for key, value in vars(config).items():
        print(f"  --{key}: {value}")
    train(config)
