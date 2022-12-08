from models.vit import ViT
from models.fashion_mynet import MyNet
from models.resnet18_pretrained import resnet18_pretrained
from models.convmixer import ConvMixer

import torch
from torch import nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import time

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Data
## FashionMNIST, CIFAR10, ImageNet
img_list = ["FashionMNIST", "CIFAR10", "ImageNet"]

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# DataLoader
batch_size = 8
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Model
## Change Whatever you want. Don't forget "to(device)"
model = ConvMixer(dim=256, depth=8, patch_size=2, kernel_size=5, n_classes=10).to(device)
# model = resnet18_pretrained.to(device)
print(model)

# Hyperparemeters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.005)
lambda1 = lambda epoch: 0.85 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

epochs = 25
for t in range(epochs):
    print(f"EPOCH {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    scheduler.step()
    test(test_dataloader, model, loss_fn)

# Save the model
PATH = './weights'
torch.save(model, f'{PATH}/{model.__class__.__name__}_{time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))}.pt')