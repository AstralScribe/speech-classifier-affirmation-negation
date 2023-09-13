import glob
from typing import List, Tuple

import lightning.pytorch as pl
import models.model as model
import pandas as pd
import torch
import torch.nn.functional as F
import utils.features
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy

logger = TensorBoardLogger("tb_logs/", name="Audio Classifier")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=2,
    dirpath="out/",
    filename="audio-classifier-{epoch:02d}-val_loss{val/loss:.2f}",
)


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
        acc = accuracy(pred, label, task="multiclass", num_classes=2)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)



        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx):
        features, label = batch
        pred = self(features)
        loss = F.cross_entropy(pred, label)
        acc = accuracy(pred, label, task="multiclass", num_classes=2)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
    
    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.LR)


if __name__ == "__main__":
    files = glob.glob("data/*.wav")
    split = 0.3
    epochs = 10
    train, test = train_test_split(files, shuffle=True, train_size=split)

    train_dataset = AudioDataset(train)
    test_dataset = AudioDataset(test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=2, 
        shuffle=False, 
        drop_last=False, 
        num_workers=2
    )

    trainer = Trainer(
        max_epochs=100,
        accumulate_grad_batches=1,
        limit_val_batches=2,
        callbacks=[checkpoint_callback],
        # logger=logger,
    )

    classifier = AudioClassifier(learning_rate=1e-5, in_channels=162)

    trainer.fit(classifier, train_loader, test_loader)
