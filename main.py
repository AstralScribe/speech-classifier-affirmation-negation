import glob
import os
from datetime import datetime

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from trainer import AudioClassifier, AudioDataset

TRAINING_FILE_PATH = "data/"
SPLIT = 0.2
MAX_EPOCHS = 32
WORKERS = 8
BATCH_SIZE = 16
VAL_BATCH_LIMIT = 8
LEARNING_RATE = 1e-5

now = datetime.now().strftime("%Y%m%d%H%M%S")

if not os.path.exists("./out"):
    os.mkdir("./out")
os.mkdir(f"./out/model_{now}")


checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=2,
    dirpath=f"out/model_{now}/",
    filename="audio-classifier-{epoch:02d}",
)

files = glob.glob(f"{TRAINING_FILE_PATH}/*.wav")
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
    limit_val_batches=VAL_BATCH_LIMIT,
    callbacks=[checkpoint_callback],
    # logger=logger,
)

classifier = AudioClassifier(learning_rate=LEARNING_RATE)

trainer.fit(classifier, train_loader, test_loader)