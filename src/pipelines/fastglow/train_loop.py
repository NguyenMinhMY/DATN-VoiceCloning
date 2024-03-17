import os
import time

import torch
import torch.multiprocessing
import wandb
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


from src.spk_embedding.StyleEmbedding import StyleEmbedding
from src.utility.storage_config import MODELS_DIR
from torch.utils.data.dataloader import DataLoader
from src.utility.utils import get_most_recent_checkpoint
from src.utility.utils import plot_progress_spec, clip_grad_norm_
from src.utility.warmup_scheduler import WarmupScheduler
from src.spk_embedding.StyleEmbedding import StyleEmbedding
from src.tts.models.fastglow.FastGlow2 import FastGlow2


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, energy, pitch, utterance condition, language_id
    return (
        pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
        torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
        pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
        torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
        pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
        pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
        None,
        torch.stack([datapoint[7] for datapoint in batch]),
    )


def train_loop(
    net: FastGlow2,
    train_dataset,
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
    init_act_norm_steps=10,
    phase_1_steps=100000,
    phase_2_steps=100000,
    use_wandb=False,
    enable_autocast=True,
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
        phase_1_steps: how many steps to train before using any of the cycle objectives
        phase_2_steps: how many steps to train using the cycle objectives
        path_to_embed_model: path to the pretrained embedding function
    """

    os.makedirs(save_directory, exist_ok=True)

    steps = phase_1_steps + phase_2_steps
    net = net.to(device)

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

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1.0e-06, weight_decay=0.0)
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)
    scaler = GradScaler()
    step_counter = 0
    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        net.load_state_dict(check_dict["model"])
        step_counter = check_dict["step_counter"]
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            scheduler.load_state_dict(check_dict["scheduler"])
            scaler.load_state_dict(check_dict["scaler"])

    net.train()

    best_train_loss = float("inf")
    best_cycle_loss = float("inf")

    print("Starting training")
    for step_counter in range(step_counter + 1, steps + 1):
        start_time = time.time()
        batch_counter = 0

        train_losses_this_epoch = list()
        l1_losses_this_epoch = list()
        mle_losses_this_epoch = list()
        duration_losses_this_epoch = list()
        cycle_losses_this_epoch = list()
        for batch in tqdm(train_loader):
            if batch_counter < init_act_norm_steps:
                net.unlock_act_norm_layers()
                with torch.no_grad():
                    style_embedding_of_gold, _ = style_embedding_function(
                        batch_of_spectrograms=batch[2].to(device),
                        batch_of_spectrogram_lengths=batch[3].to(device),
                        return_all_outs=True,
                    )
                    _ = net(
                        text_tensors=batch[0].to(device),
                        text_lengths=batch[1].to(device),
                        gold_speech=batch[2].to(device),
                        speech_lengths=batch[3].to(device),
                        utterance_embedding=style_embedding_of_gold.detach(),
                        lang_ids=batch[7].to(device),
                        return_mels=True,
                    )
                net.lock_act_norm_layers()
            else:
                with autocast(enabled=enable_autocast, cache_enabled=False):
                    style_embedding_function.eval()
                    style_embedding_of_gold, out_list_gold = style_embedding_function(
                        batch_of_spectrograms=batch[2].to(device),
                        batch_of_spectrogram_lengths=batch[3].to(device),
                        return_all_outs=True,
                    )
                    (
                        train_loss,
                        output_spectrograms,
                        l1_loss,
                        mle_loss,
                        duration_loss,
                    ) = net(
                        text_tensors=batch[0].to(device),
                        text_lengths=batch[1].to(device),
                        gold_speech=batch[2].to(device),
                        speech_lengths=batch[3].to(device),
                        utterance_embedding=style_embedding_of_gold.detach(),
                        lang_ids=batch[7].to(device),
                        return_mels=True,
                    )
                    style_embedding_function.gst.ref_enc.gst.train()
                    (
                        style_embedding_of_predicted,
                        out_list_predicted,
                    ) = style_embedding_function(
                        batch_of_spectrograms=output_spectrograms,
                        batch_of_spectrogram_lengths=batch[3].to(device),
                        return_all_outs=True,
                    )

                    cycle_dist = 0
                    for out_gold, out_pred in zip(out_list_gold, out_list_predicted):
                        # essentially feature matching, as is often done in vocoder training,
                        # since we're essentially dealing with a discriminator here.
                        cycle_dist = cycle_dist + torch.nn.functional.l1_loss(
                            out_pred, out_gold.detach()
                        )

                    train_losses_this_epoch.append(train_loss.item())
                    l1_losses_this_epoch.append(l1_loss.item())
                    mle_losses_this_epoch.append(mle_loss.item())
                    duration_losses_this_epoch.append(duration_loss.item())
                    cycle_losses_this_epoch.append(cycle_dist.item())

                    if step_counter <= phase_1_steps:
                        # ===============================================
                        # =        PHASE 1: no cycle objective          =
                        # ===============================================
                        pass
                    else:
                        # ================================================
                        # = PHASE 2:     cycle objective is added        =
                        # ================================================
                        if not cycle_dist.isnan():
                            train_loss = train_loss + cycle_dist

                style_embedding_function.zero_grad()
                optimizer.zero_grad()
                scaler.scale(train_loss).backward()

                scaler.unscale_(optimizer)
                clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            batch_counter += 1
            
        net.eval()
        style_embedding_function.eval()

        train_loss_epoch = (
            sum(train_losses_this_epoch) / len(train_losses_this_epoch)
            if len(train_losses_this_epoch) > 0
            else 0.0
        )
        l1_loss_epoch = (
            sum(l1_losses_this_epoch) / len(l1_losses_this_epoch)
            if len(l1_losses_this_epoch) > 0
            else 0.0
        )
        mle_loss_epoch = (
            sum(mle_losses_this_epoch) / len(mle_losses_this_epoch)
            if len(mle_losses_this_epoch) > 0
            else 0.0
        )
        duration_loss_epoch = (
            sum(duration_losses_this_epoch) / len(duration_losses_this_epoch)
            if len(duration_losses_this_epoch) > 0
            else 0.0
        )
        cycle_loss_epoch = (
            sum(cycle_losses_this_epoch) / len(cycle_losses_this_epoch)
            if len(cycle_losses_this_epoch) > 0
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
            if train_loss_epoch < best_train_loss:
                best_train_loss = train_loss_epoch
                torch.save(
                    {
                        "model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step_counter": step_counter,
                        "default_emb": default_embedding,
                    },
                    os.path.join(save_directory, "checkpoint_best_train_loss.pt"),
                )

            # Save the best model based on the cycle loss
            if cycle_loss_epoch < best_cycle_loss and cycle_loss_epoch != 0.0:
                best_cycle_loss = cycle_loss_epoch
                torch.save(
                    {
                        "model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step_counter": step_counter,
                        "default_emb": default_embedding,
                    },
                    os.path.join(save_directory, "checkpoint_best_cycle_loss.pt"),
                )

            # Save the lastest model
            torch.save(
                {
                    "model": net.state_dict(),
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
        print(
            "Training Loss: {:.5f} - Mel Loss: {:.5f} - MLE Loss: {:.5f} - Duration Loss: {:.5f} - Cycle Loss: {:.5f}".format(
                train_loss_epoch,
                l1_loss_epoch,
                mle_loss_epoch,
                duration_loss_epoch,
                cycle_loss_epoch,
            )
        )

        print(
            "Time elapsed:  {} Minutes".format(round((time.time() - start_time) / 60))
        )

        if use_wandb:
            wandb.log(
                {
                    "Training_loss": train_loss_epoch,
                    "L1_loss": l1_loss_epoch,
                    "MLE_loss": mle_loss_epoch,
                    "Duration_loss": duration_loss_epoch,
                    "Cycle_loss": cycle_loss_epoch,
                    "Steps": step_counter,
                    # "progress_plot": wandb.Image(path_to_most_recent_plot),
                }
            )

        net.train()
