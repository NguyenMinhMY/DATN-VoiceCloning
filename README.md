# Voice Cloning

The demand for Text-To-Speech (TTS) technology is increasing with the advancement of artificial intelligence. To meet the current needs, lots of research and experimentation have been conducted to enhance TTS capabilities, resulting in the development of a baseline model showcasing substantial proficiency in converting text to speech. However, challenges remain in the domain of voice cloning, where generated voices often lack fidelity to the originals and zero-shot learning methods have shown limited effectiveness with unseen datasets. To address these challenges, this project introduces DGSpeech, which is built on the FastSpeech2 architecture. DGSpeech incorporates mix-style layer normalization and flow-based post-net enhancements to improve voice cloning performance. Experimental evaluations compare DGSpeech with the baseline technique, which relies on FastSpeech2. Results indicate that DGSpeech achieves a Mean Opinion Score (MOS) of 3.71, surpassing the baseline technique's MOS of 3.37 in terms of naturalness. However, DGSpeech exhibits a slightly lower performance in Word Error Rate (WER) compared to the baseline. Further analysis reveals that DGSpeech generally outperforms the baseline model in cosine similarity metrics, particularly among speakers included in the training data. Nonetheless, challenges remain with unseen speakers, highlighting opportunities for future research and refinement.

## Project structure

Here is the complete and organized project structure for this repository:

```bash
project-root-directory/
├── data/                  # Includes dataset
│   ├── librispeech/
├── evaluation/            # Includes samples for evaluation
│   └── secs/              # Metrics to evaluate
│       ├── audio_samples/ # Includes referenced audio
│       └── eval_sentences.txt # Includes the input text for model evaluation
├── src/                   # Includes source code for this project
├── synthesized/           # Path for output synthesized audio folder
├── weights/               # Includes pre-trained weights and output weights for model
├── requirements.txt       # Project dependencies
├── build_data_cache.py
├── evaluation.py
├── inference.py
└── train.py
```

## Prerequisite

This repository is designed to run on Ubuntu 22.04.3 with Python 3.10. If you are using a different operating system, such as Windows or macOS, it may not work as expected.

To install this repository locally, follow these steps and resolve any errors that may occur :D

```bash
# create virtual environment
python -m venv ./venv
source ./venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### Download model weights and dataset

- Download the dataset from [here](https://www.kaggle.com/datasets/hung1578/librispeechtrain100-test-clean/data) and place it in the `data` folder.
- Download the weights of these below model in [here](https://www.kaggle.com/datasets/hung1578/voice-cloning-thesis-2024-checkpoint) and place them in the `weights` folder.

After setting up the folders, your project directory should look like this:

```bash
project-root-directory/
├── weights/
│   ├── aligner.pt
│   ├── embedding_function.pt
│   └── vocoder.pt
├── data/
│   └── librispeech/
│       ├── test-clean/
│       └── train-clean-100/
└── (other project files)
...
```

## Usage

### Build cache

After training the model, we need to build a cache from the dataset for faster data loading during the training process.

```bash
python build_data_cache.py --data_dir ./data/librispeech/train-clean
```

Some important arguments:

- `--data_dir` : Path to the dataset directory (e.g., ./data/librispeech/test-clean).
- `--aligner_checkpoint` : Path to the aligner checkpoint (default: ./weights/aligner.pt).
- `--cache_dir` : Path to the cache directory (default: .cache/ with the cache filename fast_train_cache.pt).

for more details, using `python build_data_cache.py -h`

### Train model

```bash
python train.py --model dgspeech --batch_size 2 --phase_1_steps 2 --phase_2_steps 2 --pretrained_checkpoint ./weights/checkpoint_models_2024-06-22_13-44-18/checkpoint.pt --enable_gpu
```

Some important arguments:

- `--model` : Name of model to train, options: fastspeech2, dgspeech.
- `--cache_dir` : Path to the cache directory (default: .cache/).
- `--learning_rate`: Learning rate, default: 1e-3.
- `--batch_size`: Batch size.
- `--warmup_steps`: Number of warmup steps, default: 0.
- `--phase_1_steps`: Number of steps for phase 1
- `--phase_2_steps`: Number of steps for phase 2
- `--embedding_function_checkpoint`: Path to the embedding function checkpoint, default: ./weights/embedding_function.pt/.
- `--pretrained_checkpoint`: Path to the model checkpoint, if not provided, the model will be trained from scratch.

for more details, using `python train.py -h`

### Evaluate model

Execute the command below to synthesize audios and make evaluation
Note: All audios must have `.wav` extension format

```bash
python evaluation.py --model dgspeech --metric wer --path_to_eval_speakers ./evaluation/wer/audio_samples --path_to_eval_sentences ./evaluation/wer/eval_sentences.txt --pretrained_checkpoint ./weights/checkpoint_models_2024-06-22_13-44-18/checkpoint.pt --enable_gpu
```

The csv file include the result's mean and standard will be generated as name `eval_results/wer_results.csv`

Some important arguments:

- `--model` : Name of model to train, options: fastspeech2, dgspeech.
- `--metric`: Metric for evaluation, options: wer, secs
- `--path_to_eval_speakers`: path to folder including audios of speakers used to evaluate
- `--path_to_eval_sentence`: path to txt file of list of sentences
- `--path_to_out_folder`: path to synthesized output folder, default: ./synthesized
- `--avocodo_checkpoint`: path to the vocoder checkpoint, default: ./weights/Avocoder.pt
- `--embedding_function_checkpoint`: Path to the embedding function checkpoint, default: ./weights/embedding_function.pt/
- `--pretrained_checkpoint`: Path to the model checkpoint, if not provided, the model will be trained from scratch.

### Inference audio

Execute the command below to synthesize audio from an input text and path to a reference audio:

```bash
python inference.py --model dgspeech --input_text "Hello world" --ref_path ./data/librispeech/test-clean/61/70968/61-70968-0000.flac --pretrained_checkpoint
```

The output file with the name `output.wav` will be generated

Some important arguments:

- `--model` : Name of model to train, options: fastspeech2, dgspeech.
- `--input_text`: Text to synthesize audio
- `--ref_path`: path to reference audio
- `--output_path`: path synthesized audio, default: output.wav
- `--path_to_out_folder`: path to synthesized output folder, default: ./synthesized
- `--avocodo_checkpoint`: path to the vocoder checkpoint, default: ./weights/Avocoder.pt
- `--embedding_function_checkpoint`: Path to the embedding function checkpoint, default: ./weights/embedding_function.pt/
- `--pretrained_checkpoint`: Path to the model checkpoint, if not provided, the model will be trained from scratch.

## Model Weights

| Model       | Training Details                              | Download Link                                                                                                   |
| ----------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Fastspeech2 | 10 epochs for phase 1, 190 epochs for phase 2 | [Download](https://www.kaggle.com/datasets/hung1578/voice-cloning-thesis-2024-checkpoint?select=fastspeech2.pt) |
| DGSpeech    | 200 epochs for phase 1                        | [Download](https://www.kaggle.com/datasets/hung1578/voice-cloning-thesis-2024-checkpoint?select=dgspeech.pt)    |

## References

This project incorporates code and model weights from the following sources:

1. **IMS-Toucan**: The code and model weights for [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan) were used in this project. IMS-Toucan is a high-quality text-to-speech synthesis system developed by the Institute for Natural Language Processing (IMS) at the University of Stuttgart. The specific model's weights utilized are the vocoder, aligner, and embedding function.

   - Repository: [DigitalPhonetics/IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan)
   - License: [MIT License](https://github.com/DigitalPhonetics/IMS-Toucan/blob/master/LICENSE)

Please ensure you comply with the licensing terms provided in the IMS-Toucan repository when using their code and models.
