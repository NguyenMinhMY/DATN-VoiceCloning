"""
Taken from ESPNet
"""

from abc import ABC

import torch
import torch.distributions as dist

from src.tts.layers.Conformer import Conformer
from src.tts.layers.DurationPredictor import DurationPredictor
from src.tts.layers.PostNet import PostNet
from src.tts.layers.VariancePredictor import VariancePredictor
from src.tts.layers.Glow import Glow

from src.tts.layers.common.LengthRegulator import LengthRegulator
from src.tts.layers.common.MixStyleLayerNorm import MixStyle

from src.utility.articulatory_features import get_feature_to_index_lookup
from src.utility.utils import initialize
from src.utility.utils import make_non_pad_mask
from src.utility.utils import make_pad_mask

from src.tts.losses.FastSpeech2Loss import FastSpeech2Loss


class FastPorta3(torch.nn.Module, ABC):
    def __init__(
        self,
        # network structure related
        idim=62,
        odim=80,
        adim=384,
        aheads=4,
        elayers=6,
        eunits=1536,
        dlayers=6,
        dunits=1536,
        postnet_layers=5,
        postnet_chans=256,
        postnet_filts=5,
        positionwise_layer_type="conv1d",
        positionwise_conv_kernel_size=1,
        use_scaled_pos_enc=True,
        use_batch_norm=True,
        encoder_normalize_before=True,
        decoder_normalize_before=True,
        encoder_concat_after=False,
        decoder_concat_after=False,
        reduction_factor=1,
        # encoder / decoder
        use_macaron_style_in_conformer=True,
        use_cnn_in_conformer=True,
        conformer_enc_kernel_size=7,
        conformer_dec_kernel_size=31,
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
        transformer_dec_dropout_rate=0.2,
        transformer_dec_positional_dropout_rate=0.2,
        transformer_dec_attn_dropout_rate=0.2,
        duration_predictor_dropout_rate=0.2,
        postnet_dropout_rate=0.5,
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
        self.reduction_factor = reduction_factor
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None
        self.dist = dist.Normal(0, 1)

        # mix style
        self.proj_norm = torch.torch.nn.Linear(utt_embed_dim, adim)
        self.mix_style_norm = MixStyle(
            p=0.5, alpha=0.1, eps=1e-6, hidden_size=self.adim
        )

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
            utt_embed=None,
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

        # define pitch predictor
        self.pitch_predictor = VariancePredictor(
            idim=adim,
            n_layers=pitch_predictor_layers,
            n_chans=pitch_predictor_chans,
            kernel_size=pitch_predictor_kernel_size,
            dropout_rate=pitch_predictor_dropout,
        )
        # continuous pitch + FastPitch style avg
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=adim,
                kernel_size=pitch_embed_kernel_size,
                padding=(pitch_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(pitch_embed_dropout),
        )

        # define energy predictor
        self.energy_predictor = VariancePredictor(
            idim=adim,
            n_layers=energy_predictor_layers,
            n_chans=energy_predictor_chans,
            kernel_size=energy_predictor_kernel_size,
            dropout_rate=energy_predictor_dropout,
        )
        # continuous energy + FastPitch style avg
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=adim,
                kernel_size=energy_embed_kernel_size,
                padding=(energy_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(energy_embed_dropout),
        )

        # define length regulator
        self.length_regulator = LengthRegulator()

        self.decoder = Conformer(
            idim=0,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            input_layer=None,
            dropout_rate=transformer_dec_dropout_rate,
            positional_dropout_rate=transformer_dec_positional_dropout_rate,
            attention_dropout_rate=transformer_dec_attn_dropout_rate,
            normalize_before=decoder_normalize_before,
            concat_after=decoder_concat_after,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style_in_conformer,
            use_cnn_module=use_cnn_in_conformer,
            cnn_module_kernel=conformer_dec_kernel_size,
            utt_embed=utt_embed_dim,
        )

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)

        self.proj_flow = torch.torch.nn.Linear(utt_embed_dim, adim)

        # define post flow
        self.post_flow = Glow(
            in_channels=odim,
            hidden_channels=192,  # post_glow_hidden
            kernel_size=5,  # post_glow_kernel_size
            dilation_rate=1,
            n_blocks=12,  # post_glow_n_blocks (original 12 in paper)
            n_layers=4,  # post_glow_n_block_layers (original 3 in paper)
            n_split=4,
            n_sqz=2,
            text_condition_channels=adim,
            share_cond_layers=False,  # post_share_cond_layers
            share_wn_layers=4,
            sigmoid_scale=False,
            condition_integration_projection=torch.nn.Conv1d(
                odim + adim, adim, 5, padding=2
            ),
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
        gold_durations,
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
            gold_durations (LongTensor): Batch of padded durations (B, Tmax + 1).
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
            gold_durations,
            gold_pitch,
            gold_energy,
            utterance_embedding=utterance_embedding,
            is_inference=False,
            lang_ids=lang_ids,
        )
        mel_outs, d_outs, p_outs, e_outs, glow_loss = fs_outs
        # modify mod part of groundtruth (speaking pace)
        if self.reduction_factor > 1:
            speech_lengths = speech_lengths.new(
                [olen - olen % self.reduction_factor for olen in speech_lengths]
            )

        # calculate loss
        mel_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
            after_outs=None,
            before_outs=mel_outs,
            d_outs=d_outs,
            p_outs=p_outs,
            e_outs=e_outs,
            ys=gold_speech,
            ds=gold_durations,
            ps=gold_pitch,
            es=gold_energy,
            ilens=text_lengths,
            olens=speech_lengths,
        )
        loss = mel_loss + duration_loss + pitch_loss + energy_loss + glow_loss

        if return_mels:
            return (
                mel_outs,
                loss,
                mel_loss,
                glow_loss,
                duration_loss,
                pitch_loss,
                energy_loss,
            )
        return loss, mel_loss, glow_loss, duration_loss, pitch_loss, energy_loss

    def _forward(
        self,
        text_tensors,
        text_lens,
        gold_speech=None,
        speech_lens=None,
        gold_durations=None,
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
            utterance_embedding=None,
            lang_ids=lang_ids,
        )  # (B, Tmax, adim)

        encoded_texts = self.mix_style_norm(
            encoded_texts, self.proj_norm(utterance_embedding.unsqueeze(1))
        )

        # forward duration predictor and variance predictors
        d_masks = make_pad_mask(text_lens, device=text_lens.device)

        if self.stop_gradient_from_pitch_predictor:
            pitch_predictions = self.pitch_predictor(
                encoded_texts.detach(), d_masks.unsqueeze(-1)
            )
        else:
            pitch_predictions = self.pitch_predictor(
                encoded_texts, d_masks.unsqueeze(-1)
            )

        if self.stop_gradient_from_energy_predictor:
            energy_predictions = self.energy_predictor(
                encoded_texts.detach(), d_masks.unsqueeze(-1)
            )
        else:
            energy_predictions = self.energy_predictor(
                encoded_texts, d_masks.unsqueeze(-1)
            )

        if is_inference:
            predicted_durations = self.duration_predictor.inference(
                encoded_texts, d_masks
            )  # (B, Tmax)
            # use prediction in inference
            embedded_pitch_curve = self.pitch_embed(
                pitch_predictions.transpose(1, 2)
            ).transpose(1, 2)
            embedded_energy_curve = self.energy_embed(
                energy_predictions.transpose(1, 2)
            ).transpose(1, 2)
            encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve
            encoded_texts = self.length_regulator(
                encoded_texts, predicted_durations, alpha
            )  # (B, Lmax, adim)
        else:
            predicted_durations = self.duration_predictor(encoded_texts, d_masks)

            # use groundtruth in training
            embedded_pitch_curve = self.pitch_embed(
                gold_pitch.transpose(1, 2)
            ).transpose(1, 2)
            embedded_energy_curve = self.energy_embed(
                gold_energy.transpose(1, 2)
            ).transpose(1, 2)
            encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve
            encoded_texts = self.length_regulator(
                encoded_texts, gold_durations
            )  # (B, Lmax, adim)

        # forward decoder
        if speech_lens is not None and not is_inference:
            if self.reduction_factor > 1:
                olens_in = speech_lens.new(
                    [olen // self.reduction_factor for olen in speech_lens]
                )
            else:
                olens_in = speech_lens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder(
            encoded_texts, h_masks, utterance_embedding
        )  # (B, Lmax, adim)
        mel_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        g_embed = self.proj_flow(utterance_embedding.unsqueeze(1))
        g_embed = g_embed.repeat(1, mel_outs.size(1), 1)

        glow_loss = None
        if is_inference:
            mel_outs = self.post_flow(
                tgt_mels=None,
                infer=is_inference,
                mel_out=mel_outs,
                encoded_texts=g_embed,
                tgt_nonpadding=None,
            )
        else:
            glow_loss = self.post_flow(
                tgt_mels=gold_speech,
                infer=is_inference,
                mel_out=mel_outs,
                encoded_texts=g_embed,
                tgt_nonpadding=h_masks,
            )

        return (
            mel_outs,
            predicted_durations,
            pitch_predictions,
            energy_predictions,
            glow_loss,
        )

    @torch.no_grad()
    def inference(
        self,
        text,
        speech=None,
        durations=None,
        pitch=None,
        energy=None,
        alpha=1.0,
        utterance_embedding=None,
        return_duration_pitch_energy=False,
        lang_id=None,
    ):
        """
        Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            durations (LongTensor, optional): Groundtruth of duration (T + 1,).
            pitch (Tensor, optional): Groundtruth of token-averaged pitch (T + 1, 1).
            energy (Tensor, optional): Groundtruth of token-averaged energy (T + 1, 1).
            alpha (float, optional): Alpha to control the speed.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.
            return_duration_pitch_energy: whether to return the list of predicted durations for nicer plotting

        Returns:
            Tensor: Output sequence of features (L, odim).

        """
        self.eval()
        x, y = text, speech
        d, p, e = durations, pitch, energy

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None
        if y is not None:
            ys = y.unsqueeze(0)
        if lang_id is not None:
            lang_id = lang_id.unsqueeze(0)

        (mel_outs, d_outs, pitch_predictions, energy_predictions, _) = self._forward(
            xs,
            ilens,
            ys,
            is_inference=True,
            alpha=alpha,
            utterance_embedding=utterance_embedding.unsqueeze(0),
            lang_ids=lang_id,
        )  # (1, L, odim)
        for phoneme_index, phoneme_vector in enumerate(xs.squeeze()):
            if phoneme_vector[get_feature_to_index_lookup()["voiced"]] == 0:
                pitch_predictions[0][phoneme_index] = 0.0
        self.train()
        if return_duration_pitch_energy:
            return mel_outs[0], d_outs[0], pitch_predictions[0], energy_predictions[0]
        return mel_outs[0]

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
