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

1. Download and move model weights (.pt) to iaslab_customer_service/cssa/static/weights/
2. Intall vncorenlp
    ```
    mkdir -p iaslab_customer_service/cssa/static/vncorenlp/models/wordsegmenter  
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar  
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab  
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr  
    mv VnCoreNLP-1.1.1.jar iaslab_customer_service/cssa/static/vncorenlp/   
    mv vi-vocab iaslab_customer_service/cssa/static/vncorenlp/models/wordsegmenter/  
    mv wordsegmenter.rdr iaslab_customer_service/cssa/static/vncorenlp/models/wordsegmenter/  
    ```
3. Download file from: https://public.vinai.io/PhoBERT_base_transformers.tar.gz or https://huggingface.co/vinai/phobert-base and extract in iaslab_customer_service/cssa/static/vinai/phobert-base/

### Train model

With GPU

```
docker build -t iaslab/cssa-gpu -f docker/iaslab_customer_service/cssa/Dockerfile-gpus .
```

### Evaluate model

Execute the command below to start app (GPU option):

```
docker run -d --gpus all -p 8000:5000 iaslab/cssa-gpu
```

### Synthesize audio

Execute the command below to start app (GPU option):

```
docker run -d --gpus all -p 8000:5000 iaslab/cssa-gpu
```

## Model Weights

TODO: weights