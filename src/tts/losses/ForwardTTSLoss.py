import torch
from torch import nn

from src.tts.losses.utils import ForwardSumLoss, MSELossMasked, L1LossMasked, SSIMLoss


class ForwardTTSLoss(nn.Module):
    """Generic configurable ForwardTTS loss."""

    def __init__(self):
        super().__init__()
        # self.spec_loss = L1LossMasked(False)
        self.spec_loss = MSELossMasked(False)
        self.dur_loss = MSELossMasked(False)
        self.aligner_loss = ForwardSumLoss()
        self.pitch_loss = MSELossMasked(False)
        self.energy_loss = MSELossMasked(False)
        self.ssim_loss = SSIMLoss()

        self.spec_loss_alpha = 1.0
        self.ssim_loss_alpha = 1.0
        self.aligner_loss_alpha = 1.0
        self.pitch_loss_alpha = 0.1
        self.dur_loss_alpha = 0.1
        self.energy_loss_alpha = 0.1
        self.binary_alignment_loss_alpha = 0.1

    @staticmethod
    def _binary_alignment_loss(alignment_hard, alignment_soft):
        """Binary loss that forces soft alignments to match the hard alignments as
        explained in `https://arxiv.org/pdf/2108.10447.pdf`.
        """
        log_sum = torch.log(
            torch.clamp(alignment_soft[alignment_hard == 1], min=1e-12)
        ).sum()
        return -log_sum / alignment_hard.sum()

    def forward(
        self,
        decoder_output,
        decoder_target,
        decoder_output_lens,
        dur_output,
        dur_target,
        pitch_output,
        pitch_target,
        energy_output,
        energy_target,
        input_lens,
        alignment_logprob=None,
        alignment_hard=None,
        alignment_soft=None,
        binary_loss_weight=None,
    ):
        loss = 0
        return_dict = {}
        if hasattr(self, "ssim_loss") and self.ssim_loss_alpha > 0:
            ssim_loss = self.ssim_loss(decoder_output, decoder_target, decoder_output_lens)
            loss = loss + self.ssim_loss_alpha * ssim_loss
            return_dict["loss_ssim"] = self.ssim_loss_alpha * ssim_loss

        if self.spec_loss_alpha > 0:
            spec_loss = self.spec_loss(
                decoder_output, decoder_target, decoder_output_lens
            )
            loss = loss + self.spec_loss_alpha * spec_loss
            return_dict["loss_spec"] = self.spec_loss_alpha * spec_loss

        if self.dur_loss_alpha > 0:
            log_dur_tgt = torch.log(dur_target.float() + 1)
            dur_loss = self.dur_loss(
                dur_output[:, :, None], log_dur_tgt[:, :, None], input_lens
            )
            loss = loss + self.dur_loss_alpha * dur_loss
            return_dict["loss_dur"] = self.dur_loss_alpha * dur_loss

        if hasattr(self, "pitch_loss") and self.pitch_loss_alpha > 0:
            pitch_loss = self.pitch_loss(
                pitch_output.transpose(1, 2), pitch_target.transpose(1, 2), input_lens
            )
            loss = loss + self.pitch_loss_alpha * pitch_loss
            return_dict["loss_pitch"] = self.pitch_loss_alpha * pitch_loss

        if hasattr(self, "energy_loss") and self.energy_loss_alpha > 0:
            energy_loss = self.energy_loss(
                energy_output.transpose(1, 2), energy_target.transpose(1, 2), input_lens
            )
            loss = loss + self.energy_loss_alpha * energy_loss
            return_dict["loss_energy"] = self.energy_loss_alpha * energy_loss

        if hasattr(self, "aligner_loss") and self.aligner_loss_alpha > 0:
            aligner_loss = self.aligner_loss(
                alignment_logprob, input_lens, decoder_output_lens
            )
            loss = loss + self.aligner_loss_alpha * aligner_loss
            return_dict["loss_aligner"] = self.aligner_loss_alpha * aligner_loss

        if self.binary_alignment_loss_alpha > 0 and alignment_hard is not None:
            binary_alignment_loss = self._binary_alignment_loss(
                alignment_hard, alignment_soft
            )
            loss = loss + self.binary_alignment_loss_alpha * binary_alignment_loss
            if binary_loss_weight:
                return_dict["loss_binary_alignment"] = (
                    self.binary_alignment_loss_alpha
                    * binary_alignment_loss
                    * binary_loss_weight
                )
            else:
                return_dict["loss_binary_alignment"] = (
                    self.binary_alignment_loss_alpha * binary_alignment_loss
                )

        return_dict["loss"] = loss
        return return_dict
