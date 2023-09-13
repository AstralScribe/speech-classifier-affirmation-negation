import torch
from torch import nn

class AudioClassifierModel(nn.Module):
    def __init__(self, in_channels):
        super(AudioClassifierModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=5, stride=1, padding=2)  
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        
        self.conv3 = nn.Conv1d(256, 128, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)
        self.maxpool4 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(16, 2)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.dropout1(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool4(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = self.dropout3(x)

        x = self.fc3(x)
        x = torch.softmax(x, dim=1)  # Applying softmax to get probabilities
        
        return x
