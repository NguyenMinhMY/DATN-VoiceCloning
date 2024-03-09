import os
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from src.datasets.formatters import *


def load_tts_samples(
    datasets: Union[List[Dict], Dict],
    eval_split=True,
    eval_split_max_size=None,
    eval_split_size=0.01,
) -> Tuple[List[List], List[List]]:
    meta_data_train_all = []
    meta_data_eval_all = [] if eval_split else None
    if not isinstance(datasets, list):
        datasets = [datasets]
    for dataset in datasets:
        formatter_name = dataset.formatter
        dataset_name = dataset.dataset_name
        root_path = dataset.path
        meta_file_train = dataset.meta_file_train
        meta_file_val = dataset.meta_file_val
        ignored_speakers = dataset.ignored_speakers
        language = dataset.language

        # setup the right data processor
        formatter = _get_formatter_by_name(formatter_name)

        # load train set
        meta_data_train = formatter(
            root_path, meta_file_train, ignored_speakers=ignored_speakers
        )
        assert (
            len(meta_data_train) > 0
        ), f" [!] No training samples found in {root_path}/{meta_file_train}"
        meta_data_train = add_extra_keys(meta_data_train, language, dataset_name)
        print(f" | > Found {len(meta_data_train)} files in {Path(root_path).resolve()}")

        # load evaluation split if set
        if eval_split:
            if meta_file_val:
                meta_data_eval = formatter(
                    root_path, meta_file_val, ignored_speakers=ignored_speakers
                )
                meta_data_eval = add_extra_keys(meta_data_eval, language, dataset_name)
            else:
                eval_size_per_dataset = (
                    eval_split_max_size // len(datasets)
                    if eval_split_max_size
                    else None
                )
                meta_data_eval, meta_data_train = split_dataset(
                    meta_data_train, eval_size_per_dataset, eval_split_size
                )
            meta_data_eval_all += meta_data_eval
        meta_data_train_all += meta_data_train

        # load attention masks for the duration predictor training
        # ???

        # set none for the next iter
        formatter = None
    return meta_data_train_all, meta_data_eval_all


def _get_formatter_by_name(name):
    """Returns the respective preprocessing function."""
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())


def add_extra_keys(metadata, language, dataset_name):
    for item in metadata:
        # add language name
        item["language"] = language
        # add unique audio name
        relfilepath = os.path.splitext(
            os.path.relpath(item["audio_file"], item["root_path"])
        )[0]
        audio_unique_name = f"{dataset_name}#{relfilepath}"
        item["audio_unique_name"] = audio_unique_name
    return metadata


def split_dataset(items, eval_split_max_size=None, eval_split_size=0.01):
    """Split a dataset into train and eval. Consider speaker distribution in multi-speaker training.

    Args:
        items (List[List]):
            A list of samples. Each sample is a list of `[audio_path, text, speaker_id]`.

        eval_split_max_size (int):
            Number maximum of samples to be used for evaluation in proportion split. Defaults to None (Disabled).

        eval_split_size (float):
            If between 0.0 and 1.0 represents the proportion of the dataset to include in the evaluation set.
            If > 1, represents the absolute number of evaluation samples. Defaults to 0.01 (1%).
    """
    speakers = [item["speaker_name"] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    if eval_split_size > 1:
        eval_split_size = int(eval_split_size)
    else:
        if eval_split_max_size:
            eval_split_size = min(
                eval_split_max_size, int(len(items) * eval_split_size)
            )
        else:
            eval_split_size = int(len(items) * eval_split_size)

    assert (
        eval_split_size > 0
    ), " [!] You do not have enough samples for the evaluation set. You can work around this setting the 'eval_split_size' parameter to a minimum of {}".format(
        1 / len(items)
    )
    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        speakers = [item["speaker_name"] for item in items]
        speaker_counter = Counter(speakers)
        while len(items_eval) < eval_split_size:
            item_idx = np.random.randint(0, len(items))
            speaker_to_be_removed = items[item_idx]["speaker_name"]
            if speaker_counter[speaker_to_be_removed] > 1:
                items_eval.append(items[item_idx])
                speaker_counter[speaker_to_be_removed] -= 1
                del items[item_idx]
        return items_eval, items
    return items[:eval_split_size], items[eval_split_size:]
