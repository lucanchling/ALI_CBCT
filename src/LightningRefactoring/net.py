import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from monai.networks.nets.densenet import (
    DenseNet
)

import pytorch_lightning as pl

# Different Network

class DN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels: int = 6,
    ) -> None:
        super(DN, self).__init__()

        self.fc0 = nn.Linear(in_channels,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_channels)

        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self,x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = x #F.softmax(self.fc3(x), dim=1)
        return output

class DNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 6,
    ) -> None:
        super(DNet, self).__init__()

        self.featNet = DenseNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=in_channels,
            growth_rate = 34,
            block_config = (6, 12, 24, 16),
        )

        self.dens = DN(
            in_channels = in_channels,
            out_channels = out_channels
        )

    def forward(self,x):
        x = self.featNet(x)
        x = self.dens(x)
        return x


# Pytorch Lightning Module

class Model(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 6,
        learning_rate: float = 0.001,
        **kwargs
    ) -> None:
        super(Model, self).__init__()

        self.save_hyperparameters()

        self.model = DNet(
            in_channels = in_channels,
            out_channels = out_channels
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input,target = batch["state"].type(torch.float32), batch["target"]
        y = self(input)
        loss = self.loss_fn(y,target)
        self.log("train_loss", loss)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        
        input,target = batch["state"].type(torch.float32), batch["target"]
        y = self(input)
        loss = self.loss_fn(y,target)
        
        for i in range(self.batch_size):
            if torch.eq(torch.argmax(y[i]),target[i]):
                good_move +=1
        running_loss +=loss.item()
        
        # running_loss /= step+1
        metric = good_move/((step+1)*self.batch_size)
        self.validation_metrics[n].append(metric)
        if self.verbose:
            print()
            print("Porcentage of good moves :",metric*100,"%")
        # metric = 1
        if metric > self.best_metrics[n]:
            self.best_metrics[n] = metric
            self.best_epoch[n] = self.global_epoch[n]
            save_path = os.path.join(self.model_dirs[n],self.model_name+"_Net_"+ self.network_scales[n]+".pth")
            torch.save(
                network.state_dict(), save_path
            )
            # data_model["best"] = save_path
            print(f"{GV.bcolors.OKGREEN}Model Was Saved ! Current Best Avg. metric: {self.best_metrics[n]} Current Avg. metric: {metric}{GV.bcolors.ENDC}")
        else:
            print(f"Model Was Not Saved ! Current Best Avg. metric: {self.best_metrics[n]} Current Avg. metric: {metric}")
        print("--------------------------------------------------------------------------")
        if self.generate_tensorboard:
            writer = self.writers[n]
            # writer.add_graph(network,input)
            writer.add_scalar("Validation accuracy",metric,self.global_epoch[n])
            writer.close()

        return metric

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)