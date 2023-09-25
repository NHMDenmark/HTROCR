import os
import albumentations as A
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

class NHMDDataset(Dataset):
    def __init__(self, path, dbtype, height=40, augment=False):
        self.path = path
        labels_file = f'gt_{dbtype}.txt'
        image_dir = 'image'
        labels_path = os.path.join(path, labels_file)
        image_path = os.path.join(path, image_dir)
        assert os.path.exists(labels_path), f"Could not find gt_{dbtype}.txt in the given path"
        assert os.path.exists(image_path) and os.path.isdir(image_path), f"Could not find `image` dir in the given path"
        with open(labels_path, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            elements = line.strip().split('\t')
            data.append(elements)
        df = pd.DataFrame(data)
        df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
        df['text'].fillna(' ', inplace=True)
        if len(df.columns) > 2:
            del df[2]
        self.data = df
        print(f'Dataset {dbtype} loaded. Size: {len(self.data)}')
        self.height = height
        self.augment = augment
        self.transform_medium, self.transform_heavy = self.__generate_transforms()
        

    def __len__(self):
        return len(self.data)
    
    def expand_img(self, img, h, w):
        expanded = Image.new("L", (w, h), color=255)
        expanded.paste(img)
        return expanded

    def read_image(self, img_path):
        if self.augment:
            medium_p = 0.8
            heavy_p = 0.02
            transform_variant = np.random.choice(['none', 'medium', 'heavy'],
                                                 p=[1 - medium_p - heavy_p, medium_p, heavy_p])
            transform = {
                'none': A.ToGray(always_apply=True),
                'medium': self.transform_medium,
                'heavy': self.transform_heavy,
            }[transform_variant]
        else:
            transform = A.NoOp(always_apply=True)
        img = Image.open(img_path).convert("L")
        w, h = img.size
        aspect_ratio = self.height / h
        new_width = round(w * aspect_ratio)
        img = img.resize((new_width, self.height))
        if new_width < 32:
            img = self.expand_img(img, self.height, 40)
        image = np.array(img)
        img_transformed = transform(image=image)['image']
        
        return img_transformed
    

    def __getitem__(self, idx):
        element = self.data.iloc[idx]
        text = element.text
        file_name = element.file_name
        img_path = os.path.join(self.path, "image", file_name)
        img = self.read_image(img_path)
        return (img, text)

    def __generate_transforms(self):
        t_medium = A.Compose([
            A.Rotate(5, border_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.InvertImg(p=0.1),

            A.OneOf([
                A.Downscale(0.25, 0.5, interpolation=cv2.INTER_LINEAR),
                A.Downscale(0.25, 0.5, interpolation=cv2.INTER_NEAREST),
            ], p=0.1),
            A.Blur(p=0.2),
            A.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise((50, 200), p=0.3),
            A.ImageCompression(0, 30, p=0.1),
            A.ToGray(always_apply=True),
        ])

        t_heavy = A.Compose([
            A.Rotate(10, border_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.InvertImg(p=0.1),

            A.OneOf([
                A.Downscale(0.1, 0.2, interpolation=cv2.INTER_LINEAR),
                A.Downscale(0.1, 0.2, interpolation=cv2.INTER_NEAREST),
            ], p=0.1),
            A.Blur((4, 9), p=0.5),
            A.Sharpen(p=0.5),
            A.RandomBrightnessContrast(0.8, 0.8, p=1),
            A.GaussNoise((1000, 10000), p=0.3),
            A.ImageCompression(0, 10, p=0.5),
            A.ToGray(always_apply=True),
        ])

        return t_medium, t_heavy