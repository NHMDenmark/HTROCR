from htrocr.nhmd_vit.data_processors.nhmd_dataset import NHMDDataset
import pytorch_lightning as pl
import htrocr.nhmd_vit.utils.device as device
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import ToTensor
from PIL import Image

    
class NHMDDataModule(pl.LightningDataModule):
    def __init__(self, data_path, tokenizer, max_len=300, train_bs=32, val_bs=16, test_bs=1, num_workers=8):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_bs = train_bs
        self.val_bs = val_bs
        self.test_bs = test_bs
        self.num_workers = num_workers


    def setup(self, mode='train'):
        if mode == 'train':
            self.train_data = NHMDDataset(self.data_path, "train")
            self.valid_data = NHMDDataset(self.data_path, "valid")
        elif mode == 'test':
            self.test_data = NHMDDataset(self.data_path, "test")

    def collate_fn(self, batch):
        images = [b[0] for b in batch]
        labels = [b[1] for b in batch]

        tokens = self.tokenizer.batch_encode_plus(labels,
                                                padding="max_length",
                                                max_length=self.max_len,
                                                return_tensors='pt',
                                                truncation=True)
        input_ids = tokens.get("input_ids")
        attention_masks = tokens.get("attention_mask")
        to_tensor = ToTensor()
        images = images = [to_tensor(im) for im in images]
        return torch.stack(images), input_ids, attention_masks

    def train_dataloader(self):
        dl = DataLoader(
            self.train_data,
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

        return dl

    def val_dataloader(self):
        dl = DataLoader(
            self.valid_data,
            batch_size=self.val_bs,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

        return dl

    def test_dataloader(self):
        dl = DataLoader(
            self.test_data,
            batch_size=self.test_bs,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

        return dl
