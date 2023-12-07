import os
import random
import time

import torch
import wandb
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import learn2learn as l2l

from src.spk_embedding.StyleEmbedding import StyleEmbedding
from src.utility.warmup_scheduler import WarmupScheduler
from src.utility.storage_config import MODELS_DIR
from src.utility.utils import get_most_recent_checkpoint
from src.utility.utils import clip_grad_norm_


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


def calc_loss(batch, net, style_embedding_function, is_phase_2=False, device="cpu"):
    with autocast(cache_enabled=False):
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
            utterance_embedding=style_embedding_of_gold.detach(),
            lang_ids=batch[8].to(device),
            return_mels=True,
        )
        style_embedding_function.train()
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

        if is_phase_2:
            train_loss = train_loss + cycle_dist

    return (
        train_loss,
        output_spectrograms,
        l1_loss,
        duration_loss,
        pitch_loss,
        energy_loss,
        cycle_dist,
    )


def fast_adapt(
    spt,
    qry,
    net,
    style_embedding_function,
    adaptation_steps=5,
    is_phase_2=False,
    device="cpu",
):
    for step in range(adaptation_steps):
        train_loss, *_ = calc_loss(
            spt, net, style_embedding_function, is_phase_2, device
        )
        net.adapt(train_loss)

    valid_loss, *sth = calc_loss(qry, net, style_embedding_function, is_phase_2, device)
    return valid_loss, *sth


def train_loop(
    net,
    dataset,
    device,
    save_directory,
    phase_1_steps,
    phase_2_steps,
    adaptation_steps,
    steps_per_save,
    path_to_checkpoint,
    path_to_embed_model=os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"),
    inner_lr=0.01,
    outer_lr=0.001,
    fine_tune=False,
    resume=False,
    warmup_steps=4000,
    use_wandb=False,
    allow_unused=True,
    allow_nograd=False,
):
    # ============
    # Preparations
    # ============

    os.makedirs(save_directory, exist_ok=True)

    steps = phase_1_steps + phase_2_steps
    net = net.to(device)
    net = l2l.algorithms.MAML(net, lr=inner_lr,allow_nograd=allow_nograd, allow_unused=allow_unused)

    style_embedding_function = StyleEmbedding().to(device)
    check_dict = torch.load(path_to_embed_model, map_location=device)
    style_embedding_function.load_state_dict(check_dict["style_emb_func"])
    style_embedding_function.eval()
    style_embedding_function.requires_grad_(False)

    optimizer = torch.optim.RAdam(
        net.parameters(), lr=0.001, eps=1.0e-06, weight_decay=0.0
    )
    scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps)
    scaler = GradScaler()

    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
        if path_to_checkpoint is None:
            raise RuntimeError(
                f"No checkpoint found that can be resumed from in {save_directory}"
            )
    step_counter = 0

    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        net.load_state_dict(check_dict["model"])
        step_counter = check_dict["step_counter"]
        if not fine_tune:
            optimizer.load_state_dict(check_dict["optimizer"])
            grad_scaler.load_state_dict(check_dict["scaler"])
            scheduler.load_state_dict(check_dict["scheduler"])
            if step_counter > steps:
                print("Desired steps already reached in loaded checkpoint.")
                return

    net.train()
    # =============================
    # Actual train loop starts here
    # =============================
    for step in tqdm(range(step_counter + 1, steps+1)):
        start_time = time.time()

        eval_losses_this_step = []
        l1_losses_this_step = []
        duration_losses_this_step = []
        pitch_losses_this_step = []
        energy_losses_this_step = []
        cycle_losses_this_step = []

        # outter loop
        optimizer.zero_grad()
        spts, qrys = dataset.next()
        meta_batch_size = len(spts)
        for spt, qry in zip(spts, qrys):
            spt = collate_and_pad(spt)
            qry = collate_and_pad(qry)

            learner = net.clone()
            (
                eval_loss,
                output_spectrograms,
                l1_loss,
                duration_loss,
                pitch_loss,
                energy_loss,
                cycle_dist,
            ) = fast_adapt(
                spt,
                qry,
                learner,
                style_embedding_function,
                is_phase_2=step_counter > phase_1_steps,
                device=device,
            )
            
            eval_losses_this_step.append(eval_loss.item())
            l1_losses_this_step.append(l1_loss.item())
            duration_losses_this_step.append(duration_loss.item())
            pitch_losses_this_step.append(pitch_loss.item())
            energy_losses_this_step.append(energy_loss.item())
            cycle_losses_this_step.append(cycle_dist.item())

            scaler.scale(eval_loss).backward()

        clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
        for p in net.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        net.eval()
        style_embedding_function.eval()

        eval_loss_step = (
            sum(eval_losses_this_step) / len(eval_losses_this_step)
            if len(eval_losses_this_step) > 0
            else 0.0
        )
        l1_loss_step = (
            sum(l1_losses_this_step) / len(l1_losses_this_step)
            if len(l1_losses_this_step) > 0
            else 0.0
        )
        duration_loss_step = (
            sum(duration_losses_this_step) / len(duration_losses_this_step)
            if len(duration_losses_this_step) > 0
            else 0.0
        )
        pitch_loss_step = (
            sum(pitch_losses_this_step) / len(pitch_losses_this_step)
            if len(pitch_losses_this_step) > 0
            else 0.0
        )
        energy_loss_step = (
            sum(energy_losses_this_step) / len(energy_losses_this_step)
            if len(energy_losses_this_step) > 0
            else 0.0
        )
        cycle_loss_step = (
            sum(cycle_losses_this_step) / len(cycle_losses_this_step)
            if len(cycle_losses_this_step) > 0
            else 0.0
        )

        print(f"\nSteps: {step}")
        print(
            "Training Loss: {} - L1 Loss: {} - Duration Loss: {} - Pitch Loss: {} - Energy Loss: {} - Cycle Loss: {}".format(
                eval_loss_step,
                l1_loss_step,
                duration_loss_step,
                pitch_loss_step,
                energy_loss_step,
                cycle_loss_step,
            )
        )

        print(
            "Time elapsed:  {} Minutes".format(round((time.time() - start_time) / 60))
        )

        if use_wandb:
            wandb.log(
                {
                    "Training_loss": eval_loss_step,
                    "L1_loss": l1_loss_step,
                    "Duration_loss": duration_loss_step,
                    "Pitch_loss": pitch_loss_step,
                    "Energy_loss": energy_loss_step,
                    "Cycle_loss": cycle_loss_step,
                    "Steps": step,
                }
            )


        if step % steps_per_save == 0:
            # Save the lastest model
            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step_counter": step,
                },
                os.path.join(save_directory, "checkpoint_lastest.pt"),
            )
            
        net.train()
