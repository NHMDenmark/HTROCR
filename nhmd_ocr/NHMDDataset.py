import os
import albumentations as A
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from TrOCREDProcessor import get_processor
from transformers import TrOCRProcessor

class NHMDDataset(Dataset):
    def __init__(self, path, dbtype, processor, max_length, augment=False):
        self.processor = processor
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
        if len(df.columns) > 2:
            del df[2]
        self.data = df
        print(f'Dataset {dbtype} loaded. Size: {len(self.data)}')
        self.max_length = max_length
        self.augment = augment
        self.transform_medium, self.transform_heavy = self.__generate_transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        element = self.data.iloc[idx]
        text = element.text
        file_name = element.file_name
        img_path = os.path.join(self.path, "image", file_name)

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
            transform = A.ToGray(always_apply=True)
        img = Image.open(img_path).convert("RGB")
        image = np.array(img)
        img_transformed = transform(image=image)['image']
        pixel_values = self.processor(img_transformed, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()
        if text == None:
            print(file_name)
        labels = self.processor.tokenizer(text,
                                        padding="max_length",
                                        max_length=self.max_length,
                                        truncation=True).input_ids
        labels = np.array(labels)

        # important: make sure that PAD tokens are ignored by the loss function
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding = {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels),
        }

        return encoding

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

# if __name__  == '__main__':
    # # Test if dataset extraction works
    # encoder_name = 'microsoft/trocr-base-handwritten'
    # decoder_name = 'xlm-roberta-base'

    # max_length = 300

    # processor = get_processor(encoder_name, decoder_name)
    # ds = NHMDDataset("../data/NHMD_train_final", "valid", processor, max_length, augment=True)
    # i=0
    # for idx, el in enumerate(ds):
    #     test = el
    #     i+=1
    #     if i == 1000:
    #         print(idx)
    #         i=0
    #sample = ds[420]
    #tensor_image = sample['pixel_values']
    #image = ((tensor_image.cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    #image = Image.fromarray(image)
    #image.save("test.png")
    
    #tokens = sample['labels']
    #tokens[tokens == -100] = processor.tokenizer.pad_token_id
    #text = processor.decode(tokens, skip_special_tokens=True)
    #print(f'{0}:\n{text}\n')
