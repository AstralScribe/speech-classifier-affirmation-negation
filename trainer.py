from typing import List, Tuple

import lightning.pytorch as pl
import models.model as model
import pandas as pd
import torch
import torch.nn.functional as F
import utils.features
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.utils.data import Dataset
from torchmetrics.functional import accuracy

torch.set_float32_matmul_precision("medium") 
logger = TensorBoardLogger("tb_logs/", name="Audio Classifier")


class AudioDataset(Dataset):
    def __init__(self, files: List) -> None:
        self.files = files
        self.data = self._file_loader()

    def _file_loader(self) -> pd.DataFrame:
        df = pd.DataFrame([self.files]).T
        df.columns = ["file_path"]
        df["features"] = df["file_path"].apply(utils.features.get_features)
        df = df.explode("features")
        df = df.reset_index(drop=True)
        df["label"] = df["file_path"].apply(self._get_label)

        return df

    def _get_label(self, text):
        words = text.split("_")
        label = words[-2]

        return label
    
    def __len__(self):
        return len(self._file_loader())

    def __getitem__(self, idx) -> Tuple[torch.Tensor]:
        features = self.data.at[idx, "features"]
        features = torch.Tensor(features)
        features = features.reshape(-1,1)
        label = self.data.at[idx, "label"]
        if label == "yes":
            label = torch.Tensor([1, 0])
        if label == "no":
            label = torch.Tensor([0, 1])

        return features, label


class AudioClassifier(pl.LightningModule):
    def __init__(self, learning_rate: float, in_channels: int) -> None:
        super().__init__()
        self.LR = learning_rate
        self.model = model.AudioClassifierModel(in_channels)

    def forward(self, features):
        return self.model(features)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx):
        features, label = batch
        # print("features:", features.shape, type(features))
        pred = self(features)
        loss = F.cross_entropy(pred, label)
        acc = accuracy(pred, label, task="binary")
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)



        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx):
        features, label = batch
        pred = self(features)
        loss = F.cross_entropy(pred, label)
        acc = accuracy(pred, label, task="binary")
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
    
    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.LR)