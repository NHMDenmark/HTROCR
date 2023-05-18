import os
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class NHMDDataset(Dataset):
    def __init__(self, path, dbtype):
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
        
    def __len__(self):
        return len(self.data)

    def read_image(self, img_path):
        img = Image.open(img_path).convert("L")
        img = img.resize((384, 384))
        return np.array(img)

    def __getitem__(self, idx):
        element = self.data.iloc[idx]
        text = element.text
        file_name = element.file_name
        img_path = os.path.join(self.path, "image", file_name)
        img = self.read_image(img_path)
        return (img, text)