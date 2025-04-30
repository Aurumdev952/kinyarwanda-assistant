import os
from collections import Counter

import noisereduce as nr
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")
model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")


def load_and_preprocess_audio(path, target_sample_rate=16000):
    # Load waveform
    waveform, sample_rate = torchaudio.load(path)

    # Downmix stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Normalize
    waveform = waveform / waveform.abs().max()

    # Denoise using noisereduce
    reduced = nr.reduce_noise(y=waveform.squeeze(0).numpy(), sr=sample_rate)
    waveform = torch.tensor(reduced).unsqueeze(0)

    # Voice Activity Detection to remove silent parts
    trimmed = torchaudio.functional.vad(waveform, sample_rate=sample_rate)

    return trimmed, sample_rate


def transcribe_cleaned_audio(waveform, sample_rate):

    # Prepare input
    inputs = processor(
        waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt"
    )
    predicted_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def clean_transcription(text):
    words = text.replace("!", " ").split()
    counter = Counter(words)
    most_common = counter.most_common(1)
    if most_common:
        word, count = most_common[0]
        if count > 5 and len(word) > 2:
            return word
    return text


# Get all .wav files in the samples directory
samples_dir = "samples"
audio_files = [file for file in os.listdir(samples_dir) if file.endswith(".wav")]

# Perform transcription on each audio file
for audio_file in audio_files:
    audio_path = os.path.join(samples_dir, audio_file)
    waveform, sr = load_and_preprocess_audio(audio_path)
    transcription = transcribe_cleaned_audio(waveform, sr)
    print("🗣️ Transcription for", audio_file, ":", transcription)
    print("✅ Cleaned:", clean_transcription(transcription))
