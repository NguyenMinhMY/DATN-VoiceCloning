import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional

from src.tts.losses._SSIMLoss import SSIMLoss as _SSIMLoss


def sequence_mask(sequence_length, max_len=None):
    """Create a sequence mask for filtering padding in a sequence tensor.

    Args:
        sequence_length (torch.tensor): Sequence lengths.
        max_len (int, Optional): Maximum sequence length. Defaults to None.

    Shapes:
        - mask: :math:`[B, T_max]`
    """
    if max_len is None:
        max_len = sequence_length.max()
    seq_range = torch.arange(
        max_len, dtype=sequence_length.dtype, device=sequence_length.device
    )
    # B x T_max
    return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = torch.nn.functional.pad(
            input=attn_logprob, pad=(1, 0), value=self.blank_logprob
        )

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                : query_lens[bid], :, : key_lens[bid] + 1
            ]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss = total_loss + loss

        total_loss = total_loss / attn_logprob.shape[0]
        return total_loss


class MSELossMasked(nn.Module):
    def __init__(self, seq_len_norm):
        super().__init__()
        self.seq_len_norm = seq_len_norm

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Shapes:
            - x: :math:`[B, T, D]`
            - target: :math:`[B, T, D]`
            - length: :math:`B`
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = (
            sequence_mask(sequence_length=length, max_len=target.size(1))
            .unsqueeze(2)
            .float()
        )
        if self.seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss = functional.mse_loss(x * mask, target * mask, reduction="none")
            loss = loss.mul(out_weights.to(loss.device)).sum()
        else:
            mask = mask.expand_as(x)
            loss = functional.mse_loss(x * mask, target * mask, reduction="sum")
            loss = loss / mask.sum()
        return loss


class L1LossMasked(nn.Module):
    def __init__(self, seq_len_norm):
        super().__init__()
        self.seq_len_norm = seq_len_norm

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Shapes:
            x: B x T X D
            target: B x T x D
            length: B
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = (
            sequence_mask(sequence_length=length, max_len=target.size(1))
            .unsqueeze(2)
            .float()
        )
        if self.seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss = functional.l1_loss(x * mask, target * mask, reduction="none")
            loss = loss.mul(out_weights.to(loss.device)).sum()
        else:
            mask = mask.expand_as(x)
            loss = functional.l1_loss(x * mask, target * mask, reduction="sum")
            loss = loss / mask.sum()
        return loss


class SSIMLoss(torch.nn.Module):
    """SSIM loss as (1 - SSIM)
    SSIM is explained here https://en.wikipedia.org/wiki/Structural_similarity
    """

    def __init__(self):
        super().__init__()
        self.loss_func = _SSIMLoss()

    def forward(self, y_hat, y, length):
        """
        Args:
            y_hat (tensor): model prediction values.
            y (tensor): target values.
            length (tensor): length of each sample in a batch for masking.

        Shapes:
            y_hat: B x T X D
            y: B x T x D
            length: B

        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        mask = sequence_mask(sequence_length=length, max_len=y.size(1)).unsqueeze(2)
        y_norm = sample_wise_min_max(y, mask)
        y_hat_norm = sample_wise_min_max(y_hat, mask)
        ssim_loss = self.loss_func(
            (y_norm * mask).unsqueeze(1), (y_hat_norm * mask).unsqueeze(1)
        )

        if ssim_loss.item() > 1.0:
            print(f" > SSIM loss is out-of-range {ssim_loss.item()}, setting it 1.0")
            ssim_loss = torch.tensor(1.0, device=ssim_loss.device)

        if ssim_loss.item() < 0.0:
            print(f" > SSIM loss is out-of-range {ssim_loss.item()}, setting it 0.0")
            ssim_loss = torch.tensor(0.0, device=ssim_loss.device)

        return ssim_loss


def sample_wise_min_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Min-Max normalize tensor through first dimension
    Shapes:
        - x: :math:`[B, D1, D2]`
        - m: :math:`[B, D1, 1]`
    """
    maximum = torch.amax(x.masked_fill(~mask, 0), dim=(1, 2), keepdim=True)
    minimum = torch.amin(x.masked_fill(~mask, np.inf), dim=(1, 2), keepdim=True)
    return (x - minimum) / (maximum - minimum + 1e-8)
