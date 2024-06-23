from src.pipelines.gst_fastspeech2.train_loop import (
    train_loop as gst_fastspeech2_train_loop,
)
from src.pipelines.dgspeech.train_loop import train_loop as dgspeech_train_loop
from src.tts.models.dgspeech.DGSpeech import DGSpeech
from src.tts.models.fastspeech2.FastSpeech2 import FastSpeech2

MODEL_OPTIONS = {
    "fastspeech2": {
        "model": FastSpeech2,
        "train_loop": gst_fastspeech2_train_loop,
    },
    "dgspeech": {
        "model": DGSpeech,
        "train_loop": dgspeech_train_loop,
    },
}

METRIC_OPTIONS = {"wer": "wer", "secs": "secs"}

ASR_CHECKPOINT = {
    "wer": "nvidia/stt_en_fastconformer_ctc_large",
    "secs": "nvidia/speakerverification_en_titanet_large",
}
