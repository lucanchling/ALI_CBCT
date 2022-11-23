from pytorch_lightning.callbacks import Callback
import torch
import matplotlib.pyplot as plt
import numpy as np
import io
import PIL.Image
from torchvision.transforms import ToTensor
from icecream import ic

def gen_plot(direction, direction_hat, scan_path):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], color='r', label='y')
    ax.quiver(0, 0, 0, direction_hat[0], direction_hat[1], direction_hat[2], color='b', label='y_hat')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.legend()
    plt.title(scan_path)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


class DirectionLogger(Callback):
    def __init__(self, log_steps=10):
        self.log_steps = log_steps
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

                scan, direction, scale, scan_path = batch
                batch_size = scan.shape[0]
                
                scan = scan.to(pl_module.device, non_blocking=True)
                direction = direction.to(pl_module.device, non_blocking=True)
                
                with torch.no_grad():
                    
                    
                    direction_hat, scale_hat = pl_module(scan)
                    
                    direction_hat = direction_hat.cpu().numpy()
                    scale_hat = scale_hat.cpu().numpy()
                    direction = direction.cpu().numpy()
                    scale = scale.cpu().numpy()
                    
                                            