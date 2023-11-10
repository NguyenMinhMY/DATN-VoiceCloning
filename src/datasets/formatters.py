import os
from glob import glob


def libri_tts(root_path, meta_files=None, ignored_speakers=None):
    """https://ai.google/tools/datasets/libri-tts/"""
    items = []
    if not meta_files:
        meta_files = glob(f"{root_path}/**/*trans.txt", recursive=True)
    else:
        if isinstance(meta_files, str):
            meta_files = [os.path.join(root_path, meta_files)]

    for meta_file in meta_files:
        _meta_file = os.path.basename(meta_file).split(".")[0]
        with open(meta_file, "r", encoding="utf-8") as ttf:
            for line in ttf:
                file_name, *text = line.split(" ")
                text = " ".join(text)
                speaker_name, chapter_id, *_ = file_name.split("-")
                _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
                wav_file = os.path.join(_root_path, file_name + ".flac")
                # ignore speakers
                if isinstance(ignored_speakers, list):
                    if speaker_name in ignored_speakers:
                        continue
                items.append(
                    {
                        "text": text,
                        "audio_file": wav_file,
                        "speaker_name": f"LTTS_{speaker_name}",
                        "root_path": root_path,
                    }
                )

    for item in items:
        assert os.path.exists(
            item["audio_file"]
        ), f" [!] wav files don't exist - {item['audio_file']}"
    return items
