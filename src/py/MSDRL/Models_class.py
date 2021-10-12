
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import os

from torch import nn
from tqdm.std import tqdm
from monai.metrics import DiceMetric
# from torchvision import models




class Brain:
    def __init__(
        self,
        network_type,
        network_nbr,
        model_dir,
        model_name,
        in_channels = 1,
        out_channels = 6,
        learning_rate = 1e-4,
        batch_size = 10,
        verbose = False
    ) -> None:
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        networks = []
        epoch_losses = []
        validation_dices = []
        models_dirs = []
        for n in range(network_nbr):
            net = network_type(
                in_channels = in_channels,
                out_channels = out_channels,
                lr = learning_rate,
            )
            net.to(self.device)
            networks.append(net)
            epoch_losses.append([])
            validation_dices.append([])
            dir_path = os.path.join(model_dir,str(n))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            models_dirs.append(dir_path)

        self.networks = networks
        self.epoch_losses = epoch_losses
        self.validation_dices = validation_dices
        self.best_dices = [0,0,0]
        self.global_epoch = [0,0,0]
        self.best_epoch = [0,0,0]

        self.model_dirs = models_dirs
        self.model_name = model_name


    def Train(self,data):
        # print(data)
        for n,network in enumerate(self.networks):
            
            self.global_epoch[n] += 1    
            if self.verbose:
                print("training network :",n)
    
            network.train()
            epoch_loss = 0
            epoch_iterator = tqdm(
                data[n], desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
            )
            for step, batch in enumerate(epoch_iterator):
                
                # print(batch["state"].size(),batch["target"].size())
                input,target = batch["state"].to(self.device),batch["target"].to(self.device)

                x = network(input)
                loss = network.loss_fn(x,target)
                loss.backward()
                network.optimizer.step()
                epoch_loss +=loss.item()
                epoch_iterator.set_description(
                    "Training (%d / %d Scans) (loss=%2.5f)" % ((step+1)*self.batch_size, self.batch_size*len(data[n]), loss)
                )

            epoch_loss /= step+1

            self.epoch_losses[n].append(epoch_loss)
            if self.verbose:
                print("Average epoch Loss :",epoch_loss)



    def Validate(self,data):
        # print(data)
        for n,network in enumerate(self.networks):
            if self.verbose:
                print("validating network :",n)

            network.eval()
            with torch.no_grad():
                running_loss = 0
                epoch_iterator = tqdm(
                    data[n], desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True
                )
                for step, batch in enumerate(epoch_iterator):
                    
                    # print(batch["state"].size(),batch["target"].size())
                    input,target = batch["state"].to(self.device),batch["target"].to(self.device)

                    x = network(input)
                    loss = network.loss_fn(x,target)

                    running_loss +=loss.item()
                    epoch_iterator.set_description(
                        "Validating (%d / %d Scans) (loss=%2.5f)" % ((step+1)*self.batch_size, self.batch_size*len(data[n]), loss)
                    )

                running_loss /= step+1
                dice = 1-running_loss

                self.validation_dices[n].append(dice)
                if self.verbose:
                    print("Validation dice :",dice)

                if dice > self.best_dices[n]:
                    self.best_dices[n] = dice
                    self.best_epoch[n] = self.global_epoch[n]
                    save_path = os.path.join(self.model_dirs[n],self.model_name+"_"+datetime.datetime.now().strftime("%Y_%d_%m")+"_E_"+str(self.best_epoch[n])+".pth")
                    torch.save(
                        network.state_dict(), save_path
                    )
                    # data_model["best"] = save_path
                    print("Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(self.best_dices[n], dice))
                else:
                    print("Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(self.best_dices[n], dice))
    

# #####################################
#  Networks
# #####################################

class DQN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 6,
        lr:float = 1e-4,
        dropout_rate: float = 0.0,
    ) -> None:
        super(DQN, self).__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.conv1 = nn.Conv3d(
            in_channels, 
            out_channels = 32, 
            kernel_size = 4, 
        )
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(
            in_channels = 32, 
            out_channels = 64, 
            kernel_size = 3,
        )
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(
            in_channels = 64, 
            out_channels = 128, 
            kernel_size = 2,
        )
        self.pool3 = nn.MaxPool3d(2)

        self.fc0 = nn.Linear(27648,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_channels)

        self.loss_fn = nn.L1Loss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self,x):
        # print(x.size())
        x=self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x=self.pool2(F.relu(self.conv2(x)))
        # print(x.size())
        x=self.pool3(F.relu(self.conv3(x)))
        # print(x.size())
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x), dim=1)
        return output



class DRLnet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        super(DRLnet, self).__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.conv1 = nn.Conv3d(
            in_channels, 
            out_channels = 32, 
            kernel_size = 4, 
        )
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(
            in_channels = 32, 
            out_channels = 42, 
            kernel_size = 3, 
        )
        self.pool2 = nn.MaxPool3d(2)

        self.fc0 = nn.Linear(115248,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 6)

        # self.pool1 = nn.MaxPool3d(kernel_size = 2)

    def forward(self,x):
        # print(x.size())
        x=self.pool1(F.relu(self.conv1(x)))
        # print(x.size())
        x=self.pool2(F.relu(self.conv2(x)))
        # print(x.size())
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output


