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
        df = pd.DataFrame(columns=['file', 'label'])
        with open(labels_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if i>10:
                break
            elements = line.split('\t')
            new_el_df = pd.DataFrame({'file': [elements[0]], 'label': [elements[1]]})
            df = pd.concat([df, new_el_df], ignore_index=True)
        self.data = df
        print(f'{dbtype} dataset size: {len(self.data)}')
        self.max_length = max_length
        self.augment = augment
        self.transform_medium, self.transform_heavy = self.__generate_transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        element = self.data.iloc[idx]
        label = element.label
        file = element.file
        img_path = os.path.join(self.path, "image", file)

        if self.augment:
            medium_p = 0.8
            heavy_p = 0.02
            transform_variant = np.random.choice(['none', 'medium', 'heavy'],
                                                 p=[1 - medium_p - heavy_p, medium_p, heavy_p])
            transform = {
                'none': None,
                'medium': self.transform_medium,
                'heavy': self.transform_heavy,
            }[transform_variant]
        else:
            transform = A.ToGray(always_apply=True)

        img = Image.open(img_path).convert("RGB")
        # print(np.array(img))
        image = np.array(img)
        img_transformed = transform(image=image)['image']
        print("post tansform")
        print(label)
        encoder_name = 'facebook/deit-tiny-patch16-224'
        decoder_name = 'pstroe/roberta-base-latin-cased'
        # processor = get_processor(encoder_name, decoder_name)
        # pixel_values = processor(image, return_tensors="pt")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        print()
        pixel_values = processor(image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.squeeze()
        labels = self.processor.tokenizer(label,
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
        print("init done")
        return encoding

    def __generate_transforms(self):
        t_medium = A.Compose([
            A.Rotate(5, border_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.InvertImg(p=0.05),

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
            A.InvertImg(p=0.05),

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

if __name__ == '__main__':

    encoder_name = 'facebook/deit-tiny-patch16-224'
    decoder_name = 'pstroe/roberta-base-latin-cased'

    max_length = 300

    processor = get_processor(encoder_name, decoder_name)
    ds = NHMDDataset("./data", "train", processor, max_length, augment=True)

    # for i in range(20):
    sample = ds[0]
    print(sample)
    tensor_image = sample['pixel_values']
    image = ((tensor_image.cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    tokens = sample['labels']
    tokens[tokens == -100] = processor.tokenizer.pad_token_id
    text = ''.join(processor.decode(tokens, skip_special_tokens=True).split())

    print(f'{0}:\n{text}\n')
    plt.imshow(image)
    plt.show()