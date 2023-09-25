from htrocr.nhmd_hybrid.data_processors.nhmd_dataset import NHMDDataset
import pytorch_lightning as pl
import htrocr.nhmd_hybrid.utils.device as device
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import ToTensor
from PIL import Image


class MaxPoolImagePad:
    def __init__(self):
        self.pool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def __call__(self, x):
        return self.pool(self.pool(x))
    
class NHMDDataModule(pl.LightningDataModule):
    def __init__(self, data_path, tokenizer, height=40, max_len=300, train_bs=32, val_bs=16, test_bs=1, num_workers=8, augment=False, do_pool=True):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.height = height
        self.max_len = max_len
        self.train_bs = train_bs
        self.val_bs = val_bs
        self.test_bs = test_bs
        self.num_workers = num_workers
        self.augment = augment
        self.do_pool = True
        self.pooler = MaxPoolImagePad()

    def setup(self, mode='train'):
        if mode == 'train':
            self.train_data = NHMDDataset(self.data_path, "train", self.height, self.augment)
            self.valid_data = NHMDDataset(self.data_path, "valid", self.height, self.augment)
        elif mode == 'test':
            self.test_data = NHMDDataset(self.data_path, "test", self.height, self.augment)

    def expand_img(self, img, h, w):
        expanded = Image.new("L", (w, h), color=255)
        expanded.paste(Image.fromarray(img))
        expanded = self.pad(expanded)

        return expanded
    


    def collate_fn(self, batch):
        images = [b[0] for b in batch]
        labels = [b[1] for b in batch]

        image_widths = [im.shape[1] for im in images]
        max_width = max(image_widths)

        attn_images = []
        for w in image_widths:
            attn_images.append([1] * w + [0] * (max_width - w))
        attn_images = (
            self.pooler(torch.tensor(attn_images).float()).long()
            if self.do_pool
            else None
        )

        h = images[0].shape[0]
        to_tensor = ToTensor()
        images = [to_tensor(self.expand_img(im, h=h, w=max_width)) for im in images]
        tokens = self.tokenizer.batch_encode_plus(labels,
                                                padding="max_length",
                                                max_length=self.max_len,
                                                return_tensors='pt',
                                                truncation=True)
        input_ids = tokens.get("input_ids")
        attention_masks = tokens.get("attention_mask")
        return torch.stack(images), input_ids, attn_images, attention_masks

    def train_dataloader(self):
        dl = DataLoader(
            self.train_data,
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        # dev = device.default_device()
        # train_dl = device.DeviceDataLoader(dl, dev)
        return dl

    def val_dataloader(self):
        dl = DataLoader(
            self.valid_data,
            batch_size=self.val_bs,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        # dev = device.default_device()
        # valid_dl = device.DeviceDataLoader(dl, dev)
        return dl

    def test_dataloader(self):
        dl = DataLoader(
            self.test_data,
            batch_size=self.test_bs,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        dev = device.default_device()
        test_dl = device.DeviceDataLoader(dl, dev)
        return test_dl
