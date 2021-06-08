import torch
from torchvision import transforms

class CNN(torch.nn.Module):
    def __init__(self,K):
        super(CNN, self).__init__()
        self.K=K
        self.conv_model=torch.nn.Sequential(
            transforms.Resize((32,64)),
            torch.nn.Conv2d(1, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.25),
            torch.nn.MaxPool2d((12,2)),
            torch.nn.Flatten(),
            torch.nn.Linear(960,128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128,self.K)
        )

    def forward(self, images):
        return self.conv_model(images)

