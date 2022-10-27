import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from net import Model

import pytorch_lightning as pl

if __name__ == "__main__":\
    
    model = Model()
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(model)
