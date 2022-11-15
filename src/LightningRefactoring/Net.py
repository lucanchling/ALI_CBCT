import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.networks.nets.efficientnet import EfficientNetBN


import pytorch_lightning as pl

# Different Network

class EffNet(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.net = EfficientNetBN('efficientnet-b0', spatial_dims=3, in_channels=1,num_classes=3)
        self.loss = nn.CosineSimilarity()

    def forward(self, x):
        return self.net(x) / torch.norm(self.net(x), dim=1, keepdim=True)

    def training_step(self, batch, batch_idx):
        scan, direction, origin = batch
        direction_hat = self(scan)
        loss = 1 - self.loss(direction_hat, direction)
        loss = torch.Tensor(loss.mean())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        scan, direction, origin = batch
        direction_hat = self(scan)
        loss = 1 - self.loss(direction_hat, direction)
        loss = torch.Tensor(loss.mean())
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        scan, direction, origin = batch
        direction_hat = self(scan)
        loss = 1 - self.loss(direction_hat, direction)
        loss = torch.Tensor(loss.mean())
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)