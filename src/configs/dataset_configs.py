from dataclasses import asdict, dataclass
from typing import List


@dataclass
class BaseDatasetConfig:
    """Base config for TTS datasets.

    Args:
        formatter (str):
            Formatter name that defines used formatter in ```TTS.tts.datasets.formatter```. Defaults to `""`.

        dataset_name (str):
            Unique name for the dataset. Defaults to `""`.

        path (str):
            Root path to the dataset files. Defaults to `""`.

        meta_file_train (str):
            Name of the dataset meta file. Or a list of speakers to be ignored at training for multi-speaker datasets.
            Defaults to `""`.

        ignored_speakers (List):
            List of speakers IDs that are not used at the training. Default None.

        language (str):
            Language code of the dataset. If defined, it overrides `phoneme_language`. Defaults to `""`.

        meta_file_val (str):
            Name of the dataset meta file that defines the instances used at validation.
    """

    formatter: str
    dataset_name: str
    path: str = ""
    meta_file_train: str = ""
    ignored_speakers: List[str] = None
    language: str = ""
    meta_file_val: str = ""
