"""
Taken from ESPNet
"""

from abc import ABC
from typing import Tuple

import torch
from torch.nn import functional as F

from src.tts.layers.Conformer import Conformer
from src.tts.layers.DurationPredictor import DurationPredictor
from src.tts.layers.VariancePredictor import VariancePredictor

from src.tts.layers.common.LengthRegulator import LengthRegulator
from src.tts.models.alignpitch.Aligner import AlignmentNetwork

from src.utility.articulatory_features import get_feature_to_index_lookup
from src.utility.utils import initialize
from src.utility.utils import make_non_pad_mask
from src.utility.utils import make_pad_mask
from src.utility.utils import maximum_path
from src.utility.utils import average_over_durations

from src.tts.losses.ForwardTTSLoss import ForwardTTSLoss


class AlignPitch(torch.nn.Module, ABC):
    """
    AlignPitch module.

    This is a module of FastSpeech 2 described in FastSpeech 2: Fast and
    High-Quality End-to-End Text to Speech. Instead of quantized pitch and
    energy, we use token-averaged value introduced in FastPitch: Parallel
    Text-to-speech with Pitch Prediction. The encoder and decoder are Conformers
    instead of regular Transformers.

        https://arxiv.org/abs/2006.04558
        https://arxiv.org/abs/2006.06873
        https://arxiv.org/pdf/2005.08100

    """

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
        max_duration=75,
        positionwise_layer_type="conv1d",
        positionwise_conv_kernel_size=1,
        use_scaled_pos_enc=True,
        use_batch_norm=True,
        encoder_normalize_before=True,
        decoder_normalize_before=True,
        encoder_concat_after=False,
        decoder_concat_after=False,
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
        # pitch predictor
        pitch_predictor_layers=5,
        pitch_predictor_chans=256,
        pitch_predictor_kernel_size=5,
        pitch_predictor_dropout=0.5,
        pitch_embed_kernel_size=1,
        pitch_embed_dropout=0.0,
        # training related
        transformer_enc_dropout_rate=0.2,
        transformer_enc_positional_dropout_rate=0.2,
        transformer_enc_attn_dropout_rate=0.2,
        transformer_dec_dropout_rate=0.2,
        transformer_dec_positional_dropout_rate=0.2,
        transformer_dec_attn_dropout_rate=0.2,
        duration_predictor_dropout_rate=0.2,
        init_type="xavier_uniform",
        init_enc_alpha=1.0,
        init_dec_alpha=1.0,
        use_masking=False,
        use_weighted_masking=True,
        # additional features
        utt_embed_dim=64,
        lang_embs=8000,
        binary_loss_warmup_epochs=50,
    ):
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.adim = adim
        self.eos = 1
        self.max_duration = max_duration
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None
        self.binary_loss_warmup_epochs = binary_loss_warmup_epochs

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

        # define aligner
        self.aligner = AlignmentNetwork(in_query_channels=odim, in_key_channels=idim)

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
        self.feat_out = torch.nn.Linear(adim, odim)

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

        # define criterion
        self.criterion = ForwardTTSLoss()

    def forward(
        self,
        text_tensors,
        text_lens,
        gold_speech,
        speech_lens,
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
            text_tensors (LongTensor): Batch of padded text vectors (B, Tmax, idim).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            gold_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_pitch (Tensor): Batch of padded token-averaged pitch (B, Lmax, 1).
            gold_energy (Tensor): Batch of padded token-averaged energy (B, Lmax, 1).
            utterance_embedding (Tensor): Batch of the speaker embedding (B, spk_embed_dim).


        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.
        """
        # Texts include EOS token from the teacher model already in this version

        # forward propagation
        outputs = self._forward(
            text_tensors=text_tensors,
            text_lens=text_lens,
            gold_speech=gold_speech,
            speech_lens=speech_lens,
            gold_pitch=gold_pitch,
            gold_energy=gold_energy,
            utterance_embedding=utterance_embedding,
            lang_ids=lang_ids,
        )

        loss_dict = self.criterion(
            decoder_output=outputs["model_outputs"],
            decoder_target=gold_speech,
            decoder_output_lens=speech_lens,
            dur_output=outputs["durations_log"],
            dur_target=outputs["o_alignment_dur"],
            pitch_output=outputs["pitch_avg"],
            pitch_target=outputs["pitch_avg_gt"],
            energy_output=outputs["energy_avg"],
            energy_target=outputs["energy_avg_gt"],
            input_lens=text_lens,
            alignment_logprob=outputs["alignment_logprob"],
            alignment_hard=outputs["alignment_mas"],
            alignment_soft=outputs["alignment_soft"],
            binary_loss_weight=None,
        )

        if return_mels:
            return loss_dict, outputs
        return loss_dict

    def _forward(
        self,
        text_tensors,
        text_lens,
        gold_speech,
        speech_lens,
        gold_pitch,
        gold_energy,
        utterance_embedding=None,
        lang_ids=None,
    ):
        if not self.multilingual_model:
            lang_ids = None

        if not self.multispeaker_model:
            utterance_embedding = None

        # forward encoder
        text_masks = self._source_mask(text_lens)  # [B, 1, Tmax]
        spec_masks = self._source_mask(speech_lens)  # [B, 1, Lmax]

        en_outs, _ = self.encoder(
            text_tensors,
            text_masks,
            utterance_embedding=utterance_embedding,
            lang_ids=lang_ids,
        )  # (B, Tmax, adim)

        # duration predictor
        d_masks = make_pad_mask(text_lens, device=text_lens.device)  # [B, Tmax]
        o_dr_log = self.duration_predictor(en_outs, d_masks)  # [B, Tmax]
        o_dr = torch.clamp(torch.exp(o_dr_log) - 1, 0, self.max_duration)  # [B, Tmax]
        o_attn = self.generate_attn(o_dr, text_masks.to(torch.float))

        # aligner
        o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas = (
            self._forward_aligner(
                text_tensors,
                gold_speech,
                text_masks,
                spec_masks,
            )
        )
        alignment_soft = alignment_soft.transpose(1, 2)
        alignment_mas = alignment_mas.transpose(1, 2)
        dr = o_alignment_dur

        o_pitch_emb, o_pitch, avg_pitch = self._forward_pitch_predictor(
            en_outs, d_masks.unsqueeze(-1), gold_pitch, dr
        )
        o_energy_emb, o_energy, avg_energy = self._forward_energy_predictor(
            en_outs, d_masks.unsqueeze(-1), gold_energy, dr
        )

        en_outs = en_outs + o_pitch_emb + o_energy_emb

        # use groundtruth in training
        en_outs = self.length_regulator(en_outs, dr)  # (B, Lmax, adim)

        # forward decoder
        zs, _ = self.decoder(
            en_outs, spec_masks, utterance_embedding
        )  # (B, Lmax, adim)
        de_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)
        attn = self.generate_attn(
            dr, text_masks.to(torch.float), spec_masks.to(torch.float)
        )

        return {
            "model_outputs": de_outs,  # [B, Lmax, odim]
            "durations_log": o_dr_log,  # [B, Tmax]
            "durations": o_dr,  # [B, Tmax]
            "attn_durations": o_attn,  # for visualization [B, Tmax, Lmax']
            "pitch_avg": o_pitch,  # [B, Tmax, 1]
            "pitch_avg_gt": avg_pitch,  # [B, Tmax, 1]
            "energy_avg": o_energy,  # [B, Tmax, 1]
            "energy_avg_gt": avg_energy,  # [B, Tmax, 1]
            "alignments": attn,  # [B, Tmax, Lmax]
            "alignment_soft": alignment_soft,  # [B, Lmax, Tmax]
            "alignment_mas": alignment_mas,  # [B, Lmax, Tmax]
            "o_alignment_dur": o_alignment_dur,  # [B, Tmax]
            "alignment_logprob": alignment_logprob,  # [B, 1, Lmax, Tmax]
            "x_mask": text_masks,  # [B, 1, Tmax]
            "y_mask": spec_masks,  # [B, 1, Lmax]
        }

    def _forward_aligner(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        x_mask: torch.IntTensor,
        y_mask: torch.IntTensor,
    ) -> Tuple[
        torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Aligner forward pass.

        1. Compute a mask to apply to the attention map.
        2. Run the alignment network.
        3. Apply MAS to compute the hard alignment map.
        4. Compute the durations from the hard alignment map.

        Args:
            x (torch.FloatTensor): Input sequence.
            y (torch.FloatTensor): Output sequence.
            x_mask (torch.IntTensor): Input sequence mask.
            y_mask (torch.IntTensor): Output sequence mask.

        Returns:
            Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                Durations from the hard alignment map, soft alignment potentials, log scale alignment potentials,
                hard alignment map.

        Shapes:
            - x: :math:`[B, T_en, C_en]`
            - y: :math:`[B, T_de, C_de]`
            - x_mask: :math:`[B, 1, T_en]`
            - y_mask: :math:`[B, 1, T_de]`

            - o_alignment_dur: :math:`[B, T_en]`
            - alignment_soft: :math:`[B, T_en, T_de]`
            - alignment_logprob: :math:`[B, 1, T_de, T_en]`
            - alignment_mas: :math:`[B, T_en, T_de]`
        """
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        alignment_soft, alignment_logprob = self.aligner(
            y.transpose(1, 2), x.transpose(1, 2), x_mask, None
        )
        alignment_mas = maximum_path(
            alignment_soft.squeeze(1).transpose(1, 2).contiguous(),
            attn_mask.squeeze(1).contiguous(),
        )
        o_alignment_dur = torch.sum(alignment_mas, -1).int()
        alignment_soft = alignment_soft.squeeze(1).transpose(1, 2)
        return o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas

    def _forward_pitch_predictor(
        self,
        o_en: torch.FloatTensor,
        x_mask: torch.IntTensor,
        pitch: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Pitch predictor forward pass.

        1. Predict pitch from encoder outputs.
        2. In training - Compute average pitch values for each input character from the ground truth pitch values.
        3. Embed average pitch values.

        Args:
            o_en (torch.FloatTensor): Encoder output.
            x_mask (torch.IntTensor): Input sequence mask.
            pitch (torch.FloatTensor, optional): Ground truth pitch values. Defaults to None.
            dr (torch.IntTensor, optional): Ground truth durations. Defaults to None.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Pitch embedding, pitch prediction.

        Shapes:
            - o_en: :math:`(B, C, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - pitch: :math:`(B, 1, T_{de})`
            - dr: :math:`(B, T_{en})`
        """

        o_pitch = self.pitch_predictor(o_en, x_mask)  # [B, T_en, 1]
        if pitch is not None:
            avg_pitch = average_over_durations(pitch.transpose(1, 2), dr).transpose(
                1, 2
            )  # [B, T_en, 1]
            o_pitch_emb = self.pitch_embed(avg_pitch.transpose(1, 2)).transpose(
                1, 2
            )  # [B, adim, T_en]
            return o_pitch_emb, o_pitch, avg_pitch
        o_pitch_emb = self.pitch_embed(o_pitch.transpose(1, 2)).transpose(
            1, 2
        )  # [B, adim, T_en]
        return o_pitch_emb, o_pitch

    def _forward_energy_predictor(
        self,
        o_en: torch.FloatTensor,
        x_mask: torch.IntTensor,
        energy: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Energy predictor forward pass.

        1. Predict energy from encoder outputs.
        2. In training - Compute average pitch values for each input character from the ground truth pitch values.
        3. Embed average energy values.

        Args:
            o_en (torch.FloatTensor): Encoder output.
            x_mask (torch.IntTensor): Input sequence mask.
            energy (torch.FloatTensor, optional): Ground truth energy values. Defaults to None.
            dr (torch.IntTensor, optional): Ground truth durations. Defaults to None.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Energy embedding, energy prediction.

        Shapes:
            - o_en: :math:`(B, C, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - pitch: :math:`(B, 1, T_{de})`
            - dr: :math:`(B, T_{en})`
        """

        o_energy = self.energy_predictor(o_en, x_mask)
        if energy is not None:
            avg_energy = average_over_durations(energy.transpose(1, 2), dr).transpose(
                1, 2
            )  # [B, T_en, 1]
            o_energy_emb = self.energy_embed(avg_energy.transpose(1, 2)).transpose(
                1, 2
            )  # [B, adim, T_en]
            return o_energy_emb, o_energy, avg_energy
        o_energy_emb = self.energy_embed(o_energy.transpose(1, 2)).transpose(
            1, 2
        )  # [B, adim, T_en]
        return o_energy_emb, o_energy

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

        (
            before_outs,
            after_outs,
            d_outs,
            pitch_predictions,
            energy_predictions,
        ) = self._forward(
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
            return after_outs[0], d_outs[0], pitch_predictions[0], energy_predictions[0]
        return after_outs[0]

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

    def _reset_parameters(self, init_type, init_enc_alpha, init_dec_alpha):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

    def generate_attn(self, dr, x_mask, y_mask=None):
        """Generate an attention mask from the durations.

        Shapes
            - dr: :math:`(B, T_{en})`
            - x_mask: :math:`(B, T_{en})`
            - y_mask: :math:`(B, T_{de})`
        """

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

        def generate_path(duration, mask):
            """
            Shapes:
                - duration: :math:`[B, T_en]`
                - mask: :math:'[B, T_en, T_de]`
                - path: :math:`[B, T_en, T_de]`
            """

            b, t_x, t_y = mask.shape
            cum_duration = torch.cumsum(duration, 1)

            cum_duration_flat = cum_duration.view(b * t_x)
            path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
            path = path.view(b, t_x, t_y)
            pad_shape = [[0, 0], [1, 0], [0, 0]]
            pad_shape = [item for sublist in pad_shape[::-1] for item in sublist]
            path = path - F.pad(path, pad_shape)[:, :-1]
            path = path * mask
            return path

        # compute decode mask from the durations
        if y_mask is None:
            y_lengths = dr.sum(1).long()
            y_lengths[y_lengths < 1] = 1
            y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(dr.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = generate_path(dr, attn_mask.squeeze(1)).to(dr.dtype)
        return attn

    def on_train_start(self, epochs_done):
        self.binary_loss_weight = (
            min(epochs_done / self.binary_loss_warmup_epochs, 1.0) * 1.0
        )
