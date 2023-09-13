import glob

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from trainer import AudioClassifier, AudioDataset


SPLIT = 0.3
MAX_EPOCHS = 10
WORKERS = 8
BATCH_SIZE = 4

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=2,
    dirpath="out/",
    filename="audio-classifier-{epoch:02d}-val_loss{val/loss:.2f}",
)

files = glob.glob("data/*.wav")
files = files[0:50]
train, test = train_test_split(files, shuffle=True, train_size=SPLIT)
train_dataset = AudioDataset(train)
test_dataset = AudioDataset(test)

print("Both dataset loaded.")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers= WORKERS
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    drop_last=False, 
    num_workers= WORKERS
)

print("Training started.")
trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accumulate_grad_batches=1,
    limit_val_batches=2,
    callbacks=[checkpoint_callback],
    # logger=logger,
)

classifier = AudioClassifier(learning_rate=1e-5, in_channels=162)

trainer.fit(classifier, train_loader, test_loader)