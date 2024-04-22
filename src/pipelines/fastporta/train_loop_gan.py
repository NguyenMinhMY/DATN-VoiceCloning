import os
import time
import random

import torch
import torch.multiprocessing
import wandb
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.spk_embedding.StyleEmbedding import StyleEmbedding
from src.utility.warmup_scheduler import WarmupScheduler
from src.utility.storage_config import MODELS_DIR
from src.utility.utils import get_most_recent_checkpoint
from src.utility.utils import clip_grad_norm_
from src.tts.models.fastporta.FastPorta import FastPorta
from src.tts.models.fastporta.SpectrogramDiscriminator import SpectrogramDiscriminator
from src.datasets.fastspeech_dataset import FastSpeechDataset


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition, language_id, speaker_id
    return (
        pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
        torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
        pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
        torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
        pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
        pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
        pad_sequence([datapoint[6] for datapoint in batch], batch_first=True),
        None,
        torch.stack([datapoint[8] for datapoint in batch]),
        [datapoint[9] for datapoint in batch],
    )


def get_random_utterance_conditions(dataset: FastSpeechDataset, spk_ids: list):
    mels = list()
    lengths = list()
    for id in spk_ids:
        datapoint = random.choice(dataset.datapoints_of_ids[id])
        mels.append(datapoint[2])
        lengths.append(datapoint[3])
    return pad_sequence(mels, batch_first=True), torch.stack(lengths).squeeze(1)


def train_loop(
    net: FastPorta,
    train_dataset: FastSpeechDataset,
    device,
    save_directory,
    batch_size=32,
    steps_per_save=500,
    lang="en",
    lr=0.0001,
    warmup_steps=4000,
    path_to_checkpoint=None,
    path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
    fine_tune=False,
    resume=False,
    steps=100000,
    use_wandb=False,
    enable_autocast=True,
    is_parallel=False,
):
    """
    Args:
        resume: whether to resume from the most recent checkpoint
        warmup_steps: how long the learning rate should increase before it reaches the specified value
        lr: The initial learning rate for the optimiser
        path_to_checkpoint: reloads a checkpoint to continue training from there
        fine_tune: whether to load everything from a checkpoint, or only the model parameters
        lang: language of the synthesis
        net: Model to train
        train_dataset: Pytorch Dataset Object for train data
        device: Device to put the loaded tensors on
        save_directory: Where to save the checkpoints
        batch_size: How many elements should be loaded at once
        epochs_per_save: how many epochs to train in between checkpoints
        steps: how many steps to train
        path_to_embed_model: path to the pretrained embedding function
    """

    os.makedirs(save_directory, exist_ok=True)

    net = net.to(device)
    discriminator = SpectrogramDiscriminator().to(device)

    style_embedding_function = StyleEmbedding().to(device)
    check_dict = torch.load(path_to_embed_model, map_location=device)
    style_embedding_function.load_state_dict(check_dict["style_emb_func"])
    style_embedding_function.eval()
    style_embedding_function.requires_grad_(False)

    # torch.multiprocessing.set_sharing_strategy("file_system")
    train_loader = DataLoader(
        batch_size=batch_size,
        dataset=train_dataset,
        drop_last=True,
        num_workers=1,
        pin_memory=False,
        shuffle=True,
        prefetch_factor=1,
        collate_fn=collate_and_pad,
        persistent_workers=False,
    )

    optimizer = torch.optim.Adam(
        list(net.parameters()) + list(discriminator.parameters()), lr=lr
    )
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)
    scaler = GradScaler()
    step_counter = 0
    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        net.load_state_dict(check_dict["model"])
        discriminator.load_state_dict(check_dict["discriminator"])
        step_counter = check_dict["step_counter"]
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            scheduler.load_state_dict(check_dict["scheduler"])
            scaler.load_state_dict(check_dict["scaler"])

    net.train()
    discriminator.train()

    best_train_loss = float("inf")

    for step_counter in range(step_counter + 1, steps + 1):
        start_time = time.time()

        epoch_losses = {
            key: list()
            for key in [
                "Train Loss",
                "Mel Loss",
                "Glow Loss",
                "Duration Loss",
                "Pitch Loss",
                "Energy Loss",
                "Discriminator Loss",
                "Generator Loss",
            ]
        }

        for batch in tqdm(train_loader):
            with autocast(enabled=enable_autocast, cache_enabled=False):
                if is_parallel:
                    batch_of_ref_utterances, batch_of_ref_utterance_lengths = (
                        batch[2],
                        batch[3],
                    )
                else:
                    batch_of_ref_utterances, batch_of_ref_utterance_lengths = (
                        get_random_utterance_conditions(train_dataset, batch[9])
                    )

                style_embedding = style_embedding_function(
                    batch_of_spectrograms=batch_of_ref_utterances.to(device),
                    batch_of_spectrogram_lengths=batch_of_ref_utterance_lengths.to(
                        device
                    ),
                )
                (
                    output_spectrograms,
                    train_loss,
                    mel_loss,
                    glow_loss,
                    duration_loss,
                    pitch_loss,
                    energy_loss,
                ) = net(
                    text_tensors=batch[0].to(device),
                    text_lengths=batch[1].to(device),
                    gold_speech=batch[2].to(device),
                    speech_lengths=batch[3].to(device),
                    gold_durations=batch[4].to(device),
                    gold_energy=batch[5].to(device),
                    gold_pitch=batch[6].to(device),
                    utterance_embedding=style_embedding,
                    lang_ids=batch[8].to(device),
                    return_mels=True,
                )

                discriminator_loss, generator_loss = calc_gan_outputs(
                    real_spectrograms=batch[2].to(device),
                    fake_spectrograms=output_spectrograms,
                    spectrogram_lengths=batch[3].to(device),
                    discriminator=discriminator,
                )
                if not torch.isnan(discriminator_loss):
                    train_loss = train_loss + discriminator_loss
                if not torch.isnan(generator_loss):
                    train_loss = train_loss + generator_loss

                epoch_losses["Train Loss"].append(train_loss.item())
                epoch_losses["Mel Loss"].append(mel_loss.item())
                epoch_losses["Glow Loss"].append(glow_loss.item())
                epoch_losses["Duration Loss"].append(duration_loss.item())
                epoch_losses["Pitch Loss"].append(pitch_loss.item())
                epoch_losses["Energy Loss"].append(energy_loss.item())
                epoch_losses["Discriminator Loss"].append(discriminator_loss.item())
                epoch_losses["Generator Loss"].append(generator_loss.item())

            optimizer.zero_grad()
            scaler.scale(train_loss).backward()

            scaler.unscale_(optimizer)
            clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        net.eval()
        discriminator.eval()
        style_embedding_function.eval()

        epoch_loss = {}
        for key, value in epoch_losses.items():
            epoch_loss[key] = (
                sum(epoch_losses[key]) / len(epoch_losses[key])
                if len(epoch_losses[key]) > 0
                else 0.0
            )

        if step_counter % steps_per_save == 0 and step_counter != 1:
            default_embedding = style_embedding_function(
                batch_of_spectrograms=train_dataset[0][2].unsqueeze(0).to(device),
                batch_of_spectrogram_lengths=train_dataset[0][3]
                .unsqueeze(0)
                .to(device),
            ).squeeze()

            # Save the best model based on the train loss
            if epoch_loss["Train Loss"] < best_train_loss:
                best_train_loss = epoch_loss["Train Loss"]
                torch.save(
                    {
                        "model": net.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step_counter": step_counter,
                        "default_emb": default_embedding,
                    },
                    os.path.join(save_directory, "checkpoint_best_train_loss.pt"),
                )

            # Save the lastest model
            torch.save(
                {
                    "model": net.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step_counter": step_counter,
                    "default_emb": default_embedding,
                },
                os.path.join(save_directory, "checkpoint_lastest.pt"),
            )

            # delete_old_checkpoints(save_directory, keep=5)

        print(f"\nSteps: {step_counter}")
        print(" - ".join([f"{key}: {value:.3f}" for key, value in epoch_loss.items()]))

        print(
            "Time elapsed:  {} Minutes".format(round((time.time() - start_time) / 60))
        )

        if use_wandb:
            wandb.log(
                {
                    **epoch_loss,
                    "Steps": step_counter,
                }
            )

        net.train()
        discriminator.train()


def calc_gan_outputs(
    real_spectrograms, fake_spectrograms, spectrogram_lengths, discriminator
):
    # we have signals with lots of padding and different shapes, so we need to extract fixed size windows first.
    fake_window, real_window = get_random_window(
        fake_spectrograms, real_spectrograms, spectrogram_lengths
    )
    # now we have windows that are [batch_size, 200, 80]
    critic_loss = discriminator.calc_discriminator_loss(
        fake_window.unsqueeze(1), real_window.unsqueeze(1)
    )
    generator_loss = discriminator.calc_generator_feedback(
        fake_window.unsqueeze(1), real_window.unsqueeze(1)
    )
    critic_loss = critic_loss
    generator_loss = generator_loss
    return critic_loss, generator_loss


def get_random_window(generated_sequences, real_sequences, lengths):
    """
    This will return a randomized but consistent window of each that can be passed to the discriminator
    Suboptimal runtime because of a loop, should not be too bad, but a fix would be nice.
    """
    generated_windows = list()
    real_windows = list()
    window_size = 100  # corresponds to 1.6 seconds of audio in real time

    for end_index, generated, real in zip(
        lengths.squeeze(), generated_sequences, real_sequences
    ):

        length = end_index
        real_spec_unpadded = real[:end_index]
        fake_spec_unpadded = generated[:end_index]
        while length < window_size:
            real_spec_unpadded = real_spec_unpadded.repeat((2, 1))
            fake_spec_unpadded = fake_spec_unpadded.repeat((2, 1))
            length = length * 2

        max_start = length - window_size
        start = random.randint(0, max_start)

        generated_windows.append(
            fake_spec_unpadded[start : start + window_size].unsqueeze(0)
        )
        real_windows.append(
            real_spec_unpadded[start : start + window_size].unsqueeze(0)
        )
    return torch.cat(generated_windows, dim=0), torch.cat(real_windows, dim=0)
