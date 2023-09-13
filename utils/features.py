from typing import List

import librosa
import numpy as np
import utils.augmentations as augmentations


def extract_features(data, sample_rate):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

    features = np.hstack((zcr, chroma_stft, mfcc, rms, mel))

    return features


def get_features(path, sample_rate:int = 24000) -> List[np.ndarray]:
    data, _ = librosa.load(path, duration=15, sr=sample_rate)
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)

    res1 = extract_features(data, sample_rate=sample_rate)

    noise_data = augmentations.noise(data)
    res2 = extract_features(noise_data, sample_rate=sample_rate)

    new_data = augmentations.stretch(data)
    data_stretch_pitch = augmentations.pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate=sample_rate)

    return [res1, res2, res3]
