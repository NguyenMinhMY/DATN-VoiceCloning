import argparse

import soundfile as sf
from glob import glob

import torch
import torchaudio
from torch.nn import Module
from numpy import trim_zeros

from src.utility.tokenizer import ArticulatoryCombinedTextFrontend as Tokenizer
from src.spk_embedding.StyleEmbedding import StyleEmbedding
from src.preprocessing.audio_processing import AudioPreprocessor
from src.tts.vocoders.hifigan.HiFiGAN import HiFiGANGenerator
from constant import MODEL_OPTIONS

def _inference(text, ref_path, acoustic_model, ap, style_embed_function, vocoder, alpha=1.0, lang="en", device="cpu"):
    wave, sr = sf.read(ref_path)
    norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)

    norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
    cached_speech = ap.audio_to_mel_spec_tensor(
        audio=norm_wave, normalize=False, explicit_sampling_rate=16000
    ).transpose(0, 1)

    cached_speech_len = torch.LongTensor([len(cached_speech)])
    cached_speech = cached_speech.unsqueeze(0)
    tokenizer = Tokenizer(language=lang)
    embed_text = tokenizer.string_to_tensor(
        text, handle_missing=False, input_phonemes=False
    )

    style_embedding = style_embed_function(
        batch_of_spectrograms=cached_speech.to(device),
        batch_of_spectrogram_lengths=cached_speech_len.to(device),
    )

    mel = acoustic_model.inference(
        text=embed_text.to(device),
        speech=None,
        alpha=alpha,
        utterance_embedding=style_embedding[0],
        return_duration_pitch_energy=False,
        lang_id=torch.Tensor([[12]])[0].to(dtype=torch.int64, device=device),
    )
    waveform = vocoder(mel.transpose(1, 0))[0]
    waveform = waveform.detach().cpu()

    return waveform

def inference(config):
    # Enable GPU if available
    if config.enable_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("GPU is not available, using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Load audio processor
    ap = AudioPreprocessor(input_sr=16000, output_sr=16000, melspec_buckets=80,
        hop_length=256,n_fft=1024,cut_silence=False,device=device)

    # Load checkpoints
    style_embed_function = StyleEmbedding().to(device)
    style_embed_check_dict = torch.load(config.embedding_function_checkpoint, map_location=device)
    style_embed_function.load_state_dict(style_embed_check_dict["style_emb_func"])
    style_embed_function.eval()
    style_embed_function.requires_grad_(False)

    vocoder = HiFiGANGenerator().to(device)
    avocodo_check_dict = torch.load(config.avocodo_checkpoint, map_location=device)
    vocoder.load_state_dict(avocodo_check_dict["generator"])
    vocoder.eval()

    # Load model
    model = MODEL_OPTIONS[config.model]["model"]().to(device)
    model_check_dict = torch.load(config.pretrained_checkpoint, map_location=device)
    model.load_state_dict(model_check_dict["model"])
    model.eval()
    model.requires_grad_(False)

    # Inference
    waveform = _inference(
        text=config.input_text,
        ref_path=config.ref_path,
        acoustic_model=model,
        ap=ap,
        style_embed_function=style_embed_function,
        vocoder=vocoder,
        alpha=config.alpha,
        lang=config.lang,
        device=device,
    )

    # Save to ouput file
    torchaudio.save(config.output_path, src=waveform, sample_rate=24000)

    return waveform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help=f"Model to inference, options: {MODEL_OPTIONS.keys()}",
        choices=MODEL_OPTIONS.keys(),
    )
    parser.add_argument(
        "-t",
        "--input_text",
        type=str,
        help="Input text",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--ref_path",
        type=str,
        help="Reference audio path",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Output path",
        default="output.wav",
    )
    parser.add_argument(
        "-se",   
        "--embedding_function_checkpoint",
        type=str,
        help="Style embedding checkpoint path",
        default="./weights/embedding_function.pt",
    )
    parser.add_argument(
        "-av",
        "--avocodo_checkpoint",
        type=str,
        help="Avocodo checkpoint path",
        default="./weights/Avocodo.pt",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="Alpha",
        default=1.0,
    )
    parser.add_argument(
        "-l",
        "--lang",
        type=str,
        help="Language",
        default="en",
    )
    parser.add_argument(
        "-g",
        "--enable_gpu",
        action="store_true",
        help="Enable GPU if available",
        default=False,
    )

    config = parser.parse_args()
    print("Configurations:")
    for key, value in vars(config).items():
        print(f"  --{key}: {value}")

    # Run inference
    inference(config)