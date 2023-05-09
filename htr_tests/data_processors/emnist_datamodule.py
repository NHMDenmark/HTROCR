from .emnist_dataset import EMNISTDataset
import pytorch_lightning as pl
import utils.device as device
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np 
from torch.utils.data import DataLoader, TensorDataset  
import torch 

class EMNISTDataModule(pl.LightningDataModule):
    def __init__(self, train_bs=32, val_bs=16, num_workers=2):
        self.train_bs = train_bs
        self.val_bs = val_bs
        self.num_workers = num_workers

    def setup(self):
        self.train_data = EMNISTDataset(number_of_sequences=4000)
        self.valid_data = EMNISTDataset(number_of_sequences=1000)

    def train_dataloader(self):
        dl = DataLoader(
            self.train_data,
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=self.num_workers,
        )
        dev = device.default_device()
        train_dl = device.DeviceDataLoader(dl, dev)
        return train_dl

    def val_dataloader(self):
        dl = DataLoader(
            self.valid_data,
            batch_size=self.val_bs,
            num_workers=self.num_workers,
        )
        dev = device.default_device()
        valid_dl = device.DeviceDataLoader(dl, dev)
        return valid_dl

    def test_dataloader(self):
        dl = DataLoader(
            self.valid_data,
            batch_size=self.val_bs,
            num_workers=self.num_workers,
        )
        dev = device.default_device()
        test_dl = device.DeviceDataLoader(dl, dev)
        return test_dl
