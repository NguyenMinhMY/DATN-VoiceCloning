import math

import torch

from src.tts.losses.DurationPredictorLoss import DurationPredictorLoss
from src.utility.utils import make_non_pad_mask


def weights_nonzero_speech(target):
    # target : B x T x mel
    # Assign weight 1.0 to all labels except for padding (id=0).
    dim = target.size(-1)
    return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)


class FastGlowLoss(torch.nn.Module):
    def __init__(self, use_masking=True, use_weighted_masking=False):
        """
        use_masking (bool):
            Whether to apply masking for padded part in loss calculation.
        use_weighted_masking (bool):
            Whether to weighted masking in loss calculation.
        """
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        self.constant_factor = 0.5 * math.log(2 * math.pi)

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(
        self,
        z_outs,
        z_mean_outs,
        z_std_outs,
        d_outs,
        p_outs,
        e_outs,
        ds,
        ps,
        es,
        ilens,
        olens,
        logdet,
    ):
        """
        Args:
            z_outs (Tensor): Batch of outputs of the latent variable z (B, Lmax, odim).
            z_mean_outs (Tensor): Batch of outputs of the latent variable z_mean (B, Lmax, odim).
            z_std_outs (Tensor): Batch of outputs of the latent variable z_std (B, Lmax, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, Tmax).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, Lmax, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, Lmax, 1).
            ds (LongTensor): Batch of durations (B, Tmax).
            ps (Tensor): Batch of target token-averaged pitch (B, Lmax, 1).
            es (Tensor): Batch of target token-averaged energy (B, Lmax, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).
            logdet (Tensor): Batch of log determinants (B,).

        Returns:
            Tensor: Flow loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            in_masks = make_non_pad_mask(ilens).to(ilens.device)
            d_outs = d_outs.masked_select(in_masks)
            ds = ds.masked_select(in_masks)

            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ilens.device)
            p_outs = p_outs.masked_select(out_masks)
            e_outs = e_outs.masked_select(out_masks)
            ps = ps.masked_select(out_masks)
            es = es.masked_select(out_masks)

        # calculate loss

        # flow loss - neg log likelihood
        pz = torch.sum(z_std_outs) + 0.5 * torch.sum(
            torch.exp(-2 * z_std_outs) * (z_outs - z_mean_outs) ** 2
        )
        log_mle = self.constant_factor + (pz - torch.sum(logdet)) / (
            torch.sum(olens) * z_outs.shape[2]
        )

        duration_loss = self.duration_criterion(d_outs, ds)  # [B, Tmax]
        pitch_loss = self.mse_criterion(p_outs, ps)  # [B, Lmax, 1]
        energy_loss = self.mse_criterion(e_outs, es)  # [B, Lmax, 1]

        # make weighted mask and apply it
        if self.use_weighted_masking:
            in_masks = make_non_pad_mask(ilens).to(ilens.device)  # [B, Tmax]
            in_weights = in_masks.float() / in_masks.sum(dim=1, keepdim=True).float()
            in_weights /= ds.size(0)  # [B, Tmax]
            duration_loss = duration_loss.mul(in_weights).masked_select(in_masks).sum()

            out_masks = make_non_pad_mask(olens).to(ilens.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= z_outs.size(0)  # [B, Lmax]
            out_masks = out_masks.unsqueeze(-1)  # [B, Lmax, 1]
            out_weights = out_weights.unsqueeze(-1)  # [B, Lmax, 1]
            pitch_loss = pitch_loss.mul(out_weights).masked_select(out_masks).sum()
            energy_loss = energy_loss.mul(out_weights).masked_select(out_masks).sum()

        return log_mle, duration_loss, pitch_loss, energy_loss
