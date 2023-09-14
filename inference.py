from utils.features import extract_features
from trainer import AudioClassifier
import librosa
import torch

CHECKPOINT = "out/audio-classifier-epoch=31-val_lossval"
DEVICE = "cuda"
model = AudioClassifier(0,162).to(DEVICE)
model.eval()

def run_inference(path, sample_rate):
    data, _ = librosa.load(path, duration=15, sr=sample_rate)
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)

    features = extract_features(data, sample_rate=sample_rate)
    features = torch.Tensor(features).to(DEVICE)
    features = features.reshape(1,-1,1)

    with torch.no_grad():
        y_hat = model(features)

    pred = y_hat.argmax()

    return pred


file = ""
sr = 24000
print(run_inference(file, sr))