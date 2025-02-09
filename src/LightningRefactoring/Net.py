import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.networks.nets.densenet import DenseNet121
import numpy as np
from icecream import ic
import pytorch_lightning as pl

# Different Network

class EffNet(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.net = DenseNet121(spatial_dims=3, in_channels=1,out_channels=4)
        self.CosSimLoss = nn.CosineSimilarity()
        self.MSELoss = nn.MSELoss(reduction='sum')

    def forward(self, x):
        y = self.net(x)
        # print(y)
        # print(y.shape)
        direction = nn.functional.normalize(y[:,:3],dim=1)
        scale = y[:,-1]
        # print(direction.shape,scale.shape)
        return direction, scale

    def training_step(self, batch, batch_idx):
        scan, direction, scale, scan_path, _ = batch
        batch_size = scan.shape[0]

        direction_hat, scale_hat = self(scan)
        # ic(direction_hat, scale)
        # scale_hat = torch.cat([scale_hat]*int(batch_size))
        
        # ic(self.MSELoss(scale_hat.float(), scale.float()))
        loss = (1 - self.CosSimLoss(direction_hat, direction))
        # Sum the loss over the batch
        loss = loss.sum() + self.MSELoss(scale_hat.float(), scale.float())
        # ic(loss)
        self.log('train_loss', loss, batch_size=batch_size)
        
        return loss

    def validation_step(self, batch, batch_idx):
        scan, direction, scale, scan_path, _ = batch
        batch_size = scan.shape[0]
        direction_hat, scale_hat = self(scan)
        # ic(direction_hat, scale_hat)
        # scale_hat = torch.cat([scale_hat]*int(batch_size))
        loss = (1 - self.CosSimLoss(direction_hat, direction))
        loss = loss.sum() + self.MSELoss(scale_hat.float(), scale.float())
        self.log('val_loss', loss, batch_size=batch_size)
        
        return loss

    def test_step(self, batch, batch_idx):
        scan, direction, scale, scan_path, _ = batch
        batch_size = scan.shape[0]

        direction_hat, scale_hat = self(scan)
        
        # scale_hat = torch.cat([scale_hat]*int(batch_size))
        
        loss = (1 - self.CosSimLoss(direction_hat, direction))
        loss = loss.sum() + self.MSELoss(scale_hat.float(), scale.float())
        self.log('test_loss', loss, batch_size=batch_size)
        
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)