import torch

from src.tts.layers.VariancePredictor import VariancePredictor


class StyleAdaptor2(torch.nn.Module):
    """
    Style Adaptor module

    This is a module which same as variance adaptor in FastSpeech2
    without having Duration Predictor and Length Regulator
    """

    def __init__(
        self,
        idim=80,
        # pitch predictor
        pitch_predictor_layers=5,
        pitch_predictor_chans=256,
        pitch_predictor_kernel_size=5,
        pitch_predictor_dropout=0.5,
        pitch_embed_kernel_size=1,
        pitch_embed_dropout=0.0,
        # energy predictor
        energy_predictor_layers=2,
        energy_predictor_chans=256,
        energy_predictor_kernel_size=3,
        energy_predictor_dropout=0.5,
        energy_embed_kernel_size=1,
        energy_embed_dropout=0.0,
    ):
        super().__init__()

        # define pitch predictor
        self.pitch_predictor = VariancePredictor(
            idim=idim,
            n_layers=pitch_predictor_layers,
            n_chans=pitch_predictor_chans,
            kernel_size=pitch_predictor_kernel_size,
            dropout_rate=pitch_predictor_dropout,
        )
        # continuous pitch + FastPitch style avg
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=idim,
                kernel_size=pitch_embed_kernel_size,
                padding=(pitch_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(pitch_embed_dropout),
        )

        # define energy predictor
        self.energy_predictor = VariancePredictor(
            idim=idim,
            n_layers=energy_predictor_layers,
            n_chans=energy_predictor_chans,
            kernel_size=energy_predictor_kernel_size,
            dropout_rate=energy_predictor_dropout,
        )
        # continuous energy + FastPitch style avg
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=idim,
                kernel_size=energy_embed_kernel_size,
                padding=(energy_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(energy_embed_dropout),
        )

    def forward(
        self,
        xs,
        padding_mask=None,
        gold_pitch=None,
        gold_energy=None,
        is_inference=False,
    ):

        pitch_predictions = energy_predictions = embedded_curve = None

        if is_inference:
            pitch_predictions = self.pitch_predictor(xs, padding_mask)
            energy_predictions = self.energy_predictor(xs, padding_mask)

            embedded_pitch_curve = self.pitch_embed(
                pitch_predictions.transpose(1, 2)
            ).transpose(
                1, 2
            )  # (B, Tmax, idim)
            embedded_energy_curve = self.energy_embed(
                energy_predictions.transpose(1, 2)
            ).transpose(
                1, 2
            )  # (B, Tmax, idim)

            embedded_curve = embedded_pitch_curve + embedded_energy_curve

        else:
            embedded_pitch_curve = self.pitch_embed(
                gold_pitch.transpose(1, 2)
            ).transpose(
                1, 2
            )  # (B, Tmax, idim)
            embedded_energy_curve = self.energy_embed(
                gold_energy.transpose(1, 2)
            ).transpose(
                1, 2
            )  # (B, Tmax, idim)

            embedded_curve = embedded_pitch_curve + embedded_energy_curve

            ys = xs - embedded_curve
            pitch_predictions = self.pitch_predictor(ys, padding_mask)
            energy_predictions = self.energy_predictor(ys, padding_mask)

        return embedded_curve, pitch_predictions, energy_predictions
