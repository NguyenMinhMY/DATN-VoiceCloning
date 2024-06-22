# Voice Cloning

TODO: abstract

## Project structure

TODO: viết kĩ model + pipeline

- `iaslab_*`: source code in this package
- `conf`: configure files
- `notebooks`: all notebooks
- `scripts`: **useful** scripts (.sh, etc.) to run
- `data`: data description, but actually data is not here (big size, not code)
- `models`: models, typical is just a links and not push to git

## Prerequisite

TODO: machine, script run requirement.txt, down model weights (aligner, toucanTTS) + dataset

## Usage

### Build cache

After training the model, we need to build a cache from the dataset for faster data loading during the training process.

```bash
python build_data_cache.py -d ./data/librispeech/train-clean
```

Some important arguments:

- `--data_dir` : Path to the dataset directory (e.g., ./data/librispeech/test-clean).
- `--aligner_checkpoint` : Path to the aligner checkpoint (default: ./weights/aligner.pt).
- `--cache_dir` : Path to the cache directory (default: .cache/ with the cache filename fast_train_cache.pt).

for more details, using `python build_data_cache.py -h`

### Train model

With GPU

```
docker build -t iaslab/cssa-gpu -f docker/iaslab_customer_service/cssa/Dockerfile-gpus .
```

### Evaluate model

Execute the command below to start app (GPU option):

```bash
docker run -d --gpus all -p 8000:5000 iaslab/cssa-gpu
```

### Synthesize audio

Execute the command below to start app (GPU option):

```bash
docker run -d --gpus all -p 8000:5000 iaslab/cssa-gpu
```

## Model Weights

TODO: weights