import math
from abc import ABC

import torch

from src.tts.layers.Conformer import Conformer
from src.tts.layers.DurationPredictor import DurationPredictor
from src.tts.layers.StyleAdaptor import StyleAdaptor
from src.tts.layers.FlowBasedDecoder import FlowBasedDecoder
from src.tts.layers.PostNet import PostNet

from src.tts.layers.common.LengthRegulator import LengthRegulator

from src.utility.articulatory_features import get_feature_to_index_lookup
from src.utility.utils import initialize
from src.utility.utils import make_non_pad_mask, make_pad_mask, maximum_path

from src.tts.losses.FastSpeech2Loss import FastSpeech2Loss


class FastGlow(torch.nn.Module, ABC):
    def __init__(
        self,
        # network structure related
        idim=62,
        odim=80,
        adim=384,
        aheads=4,
        elayers=6,
        eunits=1536,
        positionwise_conv_kernel_size=1,
        use_scaled_pos_enc=True,
        encoder_normalize_before=True,
        encoder_concat_after=False,
        # reduction_factor=1,
        # encoder
        use_macaron_style_in_conformer=True,
        use_cnn_in_conformer=True,
        conformer_enc_kernel_size=7,
        # decoder
        dec_hidden_channels=192,
        dec_kernel_size=5,
        dec_dilation_rate=1,
        dec_num_flow_blocks=12,
        dec_num_coupling_layers=4,
        dec_dropout_p=0.05,
        dec_num_splits=4,
        dec_num_squeeze=2,
        dec_sigmoid_scale=False,
        # duration predictor
        duration_predictor_layers=2,
        duration_predictor_chans=256,
        duration_predictor_kernel_size=3,
        # energy predictor
        energy_predictor_layers=2,
        energy_predictor_chans=256,
        energy_predictor_kernel_size=3,
        energy_predictor_dropout=0.5,
        energy_embed_kernel_size=1,
        energy_embed_dropout=0.0,
        stop_gradient_from_energy_predictor=False,
        # pitch predictor
        pitch_predictor_layers=5,
        pitch_predictor_chans=256,
        pitch_predictor_kernel_size=5,
        pitch_predictor_dropout=0.5,
        pitch_embed_kernel_size=1,
        pitch_embed_dropout=0.0,
        stop_gradient_from_pitch_predictor=False,
        # training related
        transformer_enc_dropout_rate=0.2,
        transformer_enc_positional_dropout_rate=0.2,
        transformer_enc_attn_dropout_rate=0.2,
        duration_predictor_dropout_rate=0.2,
        init_type="xavier_uniform",
        init_enc_alpha=1.0,
        init_dec_alpha=1.0,
        use_masking=False,
        use_weighted_masking=True,
        # additional features
        utt_embed_dim=64,
        lang_embs=8000,
    ):
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.adim = adim
        self.eos = 1
        # self.reduction_factor = reduction_factor
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None

        # define encoder
        embed = torch.nn.Sequential(
            torch.nn.Linear(idim, 100), torch.nn.Tanh(), torch.nn.Linear(100, adim)
        )
        self.encoder = Conformer(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=embed,
            dropout_rate=transformer_enc_dropout_rate,
            positional_dropout_rate=transformer_enc_positional_dropout_rate,
            attention_dropout_rate=transformer_enc_attn_dropout_rate,
            normalize_before=encoder_normalize_before,
            concat_after=encoder_concat_after,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style_in_conformer,
            use_cnn_module=use_cnn_in_conformer,
            cnn_module_kernel=conformer_enc_kernel_size,
            zero_triu=False,
            utt_embed=utt_embed_dim,
            lang_embs=lang_embs,
        )

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # define style adaptor
        self.style_adaptor = StyleAdaptor(
            adim=adim,
            # pitch predictor
            pitch_predictor_layers=pitch_predictor_layers,
            pitch_predictor_chans=pitch_predictor_chans,
            pitch_predictor_kernel_size=pitch_predictor_kernel_size,
            pitch_predictor_dropout=pitch_predictor_dropout,
            pitch_embed_kernel_size=pitch_embed_kernel_size,
            pitch_embed_dropout=pitch_embed_dropout,
            # energy predictor
            energy_predictor_layers=energy_predictor_layers,
            energy_predictor_chans=energy_predictor_chans,
            energy_predictor_kernel_size=energy_predictor_kernel_size,
            energy_predictor_dropout=energy_predictor_dropout,
            energy_embed_kernel_size=energy_embed_kernel_size,
            energy_embed_dropout=energy_embed_dropout,
        )

        # define style adaptor's project layer
        self.proj_m = torch.nn.Conv1d(adim, odim, 1)
        self.proj_s = torch.nn.Conv1d(adim, odim, 1)
        self.proj_dec = torch.nn.Conv1d(adim, odim, 1)

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        self.decoder = FlowBasedDecoder(
            in_channels=odim,
            hidden_channels=dec_hidden_channels,
            kernel_size=dec_kernel_size,
            dilation_rate=dec_dilation_rate,
            num_flow_blocks=dec_num_flow_blocks,
            num_coupling_layers=dec_num_coupling_layers,
            dropout_p=dec_dropout_p,
            num_splits=dec_num_splits,
            num_squeeze=dec_num_squeeze,
            sigmoid_scale=dec_sigmoid_scale,
            c_in_channels=utt_embed_dim,
        )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

        # define criterion
        self.criterion = FastSpeech2Loss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking
        )

    def forward(
        self,
        text_tensors,
        text_lengths,
        gold_speech,
        speech_lengths,
        gold_pitch,
        gold_energy,
        utterance_embedding,
        return_mels=False,
        lang_ids=None,
    ):
        """
        Calculate forward propagation.

        Args:
            return_mels: whether to return the predicted spectrogram
            text_tensors (LongTensor): Batch of padded text vectors (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            gold_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_pitch (Tensor): Batch of padded token-averaged pitch (B, Tmax + 1, 1).
            gold_energy (Tensor): Batch of padded token-averaged energy (B, Tmax + 1, 1).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.
        """
        # Texts include EOS token from the teacher model already in this version

        # forward propagation
        fs_outs = self._forward(
            text_tensors,
            text_lengths,
            gold_speech,
            speech_lengths,
            gold_pitch,
            gold_energy,
            utterance_embedding=utterance_embedding,
            is_inference=False,
            lang_ids=lang_ids,
        )
        before_outs, after_outs, d_outs, p_outs, e_outs, mas_outs = fs_outs
        # modify mod part of groundtruth (speaking pace)
        # if self.reduction_factor > 1:
        #     speech_lengths = speech_lengths.new(
        #         [olen - olen % self.reduction_factor for olen in speech_lengths]
        #     )

        # calculate loss
        l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
            after_outs=after_outs,
            before_outs=before_outs,
            d_outs=d_outs,
            p_outs=p_outs,
            e_outs=e_outs,
            ys=gold_speech,
            ds=mas_outs,
            ps=gold_pitch,
            es=gold_energy,
            ilens=text_lengths,
            olens=speech_lengths,
        )
        loss = l1_loss + duration_loss + pitch_loss + energy_loss

        if return_mels:
            return loss, before_outs, l1_loss, duration_loss, pitch_loss, energy_loss
        return loss, l1_loss, duration_loss, pitch_loss, energy_loss

    def _forward(
        self,
        text_tensors,
        text_lens,
        gold_speech=None,
        speech_lens=None,
        gold_pitch=None,
        gold_energy=None,
        is_inference=False,
        alpha=1.0,
        utterance_embedding=None,
        lang_ids=None,
    ):
        if not self.multilingual_model:
            lang_ids = None

        if not self.multispeaker_model:
            utterance_embedding = None

        # forward encoder
        text_masks = self._source_mask(text_lens)

        encoded_texts, _ = self.encoder(
            text_tensors,
            text_masks,
            utterance_embedding=utterance_embedding,
            lang_ids=lang_ids,
        )  # (B, Tmax, adim)

        # forward duration predictor and variance predictors
        d_masks = make_pad_mask(text_lens, device=text_lens.device)  # [B, Tmax]

        h_masks = None
        mas_durations = None

        if is_inference:
            predicted_durations = self.duration_predictor(
                encoded_texts, d_masks
            )  # (B, Tmax)
            predicted_durations = torch.clamp(predicted_durations, min=0.0)

            embedded_curve, pitch_predictions, energy_predictions = self.style_adaptor(
                xs=encoded_texts,
                padding_mask=d_masks.unsqueeze(-1),
                gold_pitch=gold_pitch,
                gold_energy=gold_energy,
                is_inference=is_inference,
            )

            encoded_texts = encoded_texts + embedded_curve  # [B, Tmax, adim]
            encoded_texts = self.length_regulator(
                encoded_texts, torch.ceil(predicted_durations).to(torch.long), alpha
            )  # [B, Lmax, adim]
            encoded_texts = self.proj_dec(encoded_texts.transpose(1, 2)).transpose(
                1, 2
            )  # [B, Lmax, odim]

            h_masks = None

        else:
            # stop gradient from the encoder
            predicted_durations = self.duration_predictor(
                encoded_texts.detach(), d_masks
            )  # (B, Tmax)

            embedded_curve, pitch_predictions, energy_predictions = self.style_adaptor(
                xs=encoded_texts.detach(),
                padding_mask=d_masks.unsqueeze(-1),
                gold_pitch=gold_pitch,
                gold_energy=gold_energy,
                is_inference=is_inference,
            )

            embedded_curve_mean = self.proj_m(embedded_curve.transpose(1, 2)).transpose(
                1, 2
            )  # [B, Tmax, odim]
            embedded_curve_std = self.proj_s(embedded_curve.transpose(1, 2)).transpose(
                1, 2
            )  # [B, Tmax, odim]

            h_masks = self._source_mask(speech_lens)  # [B, 1, Lmax]

            z, _ = self.decoder(
                gold_speech.transpose(1, 2),
                x_mask=h_masks,
                g=utterance_embedding.unsqueeze(-1),
                reverse=False,
            )
            z = z.transpose(1, 2)

            mas_durations = self._calc_duration_using_mas(
                x_s=embedded_curve_mean,
                x_m=embedded_curve_std,
                z=z,
                x_mask=self._source_mask(text_lens),
                z_mask=h_masks,
            )

            encoded_texts = encoded_texts + embedded_curve  # [B, Tmax, adim]
            encoded_texts = self.length_regulator(
                encoded_texts, mas_durations, alpha
            )  # [B, Lmax, adim]
            encoded_texts = self.proj_dec(encoded_texts.transpose(1, 2)).transpose(
                1, 2
            )  # [B, Lmax, odim]

        ys, _ = self.decoder(
            encoded_texts.transpose(1, 2),
            h_masks,
            utterance_embedding.unsqueeze(-1),
            reverse=True,
        )
        ys = ys.transpose(1, 2)  # [B, Lmax, odim]

        return (
            ys,
            None,
            predicted_durations,
            pitch_predictions,
            energy_predictions,
            mas_durations,
        )

    def _source_mask(self, ilens):
        """
        Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.

        """
        x_masks = make_non_pad_mask(ilens, device=ilens.device)
        return x_masks.unsqueeze(-2)

    def _calc_duration_using_mas(self, x_m, x_s, z, x_mask, z_mask):
        """
        Calculate duration using MAS.

        Args:
            s_m (tensor): [B, Tmax, odim]
            x_s (tensor): [B, Tmax, odim]
            z (tensor): [B, Lmax, odim]
            x_mask (tensor): [B, 1, Tmax]
            z_mask (tensor): [B, 1, Lmax]

        Returns:
            LongTensor: [B, Tmax+1]
        """

        x_m = torch.transpose(x_m, 1, 2)  # [B, odim, Tmax]
        x_s = torch.transpose(x_s, 1, 2)  # [B, odim, Tmax]
        z = torch.transpose(z, 1, 2)  # [B, odim, Lmax]

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(
            z_mask, 2
        )  # [B, Tmax, Lmax]

        with torch.no_grad():
            o_scale = torch.exp(-2 * x_s)  # [B, odim, Tmax]
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_s, [1]).unsqueeze(
                -1
            )  # [B, odim, 1]
            logp2 = torch.matmul(
                o_scale.transpose(1, 2), -0.5 * (z**2)
            )  # [B, Tmax, Lmax]
            logp3 = torch.matmul((x_m * o_scale).transpose(1, 2), z)  # [B, Tmax, Lmax]
            logp4 = torch.sum(-0.5 * (x_m**2) * o_scale, [1]).unsqueeze(
                -1
            )  # [B, odim, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [B, Tmax, Lmax]
            attn = (
                maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
            )  # [B, 1, Lmax]
        return attn.squeeze(1).sum(dim=-1).to(torch.long)  # [B, Tmax]
        # durations = attn.squeeze(1).sum(dim=-1) # [B, Tmax]
        # return torch.cat((durations, torch.unsqueeze(torch.zeros(durations.size()[0], device=durations.device), 1)), dim=1)

    def _reset_parameters(self, init_type, init_enc_alpha, init_dec_alpha):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)
