import librosa
import numpy as np


def noise(y: np.ndarray) -> np.ndarray:
    noise_amp = 0.035 * np.random.uniform() * np.amax(y)
    y = y + noise_amp * np.random.normal(size=y.shape[0])
    return y


def stretch(y: np.ndarray, rate: float = 0.8) -> np.ndarray:
    return librosa.effects.time_stretch(y=y, rate=rate)


def shift(y: np.ndarray) -> np.ndarray:
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(y, shift_range)


def pitch(y: np.ndarray, sampling_rate: int, n_steps: int = 4) -> np.ndarray:
    return librosa.effects.pitch_shift(y=y, sr=sampling_rate, n_steps=n_steps)
