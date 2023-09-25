import htrocr.nhmd_hybrid.utils.device as device
from torchvision import datasets
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as tt
import torchvision.transforms.functional as tf
import torch
from PIL import Image
from torch.utils.data import Dataset

class EMNISTDataset(Dataset):
    def __init__(self, path='./EMNIST', digits_per_sequence=5, number_of_sequences=5000):
        self.emnist_dataset = datasets.EMNIST('./EMNIST', split="digits", train=True, download=True)
        self.digits_per_sequence = digits_per_sequence
        self.number_of_sequences = number_of_sequences
        self.rand_idxs = [np.random.randint(len(self.emnist_dataset.data), size=(digits_per_sequence,)) for _ in range(self.number_of_sequences)]
        

    def __len__(self):
        return self.number_of_sequences

    def expand_img(self, img, h, w):
        expanded = Image.new("L", (w, h), color=255)
        expanded.paste(img)
        return expanded
    
    def __getitem__(self, idx):
        rand_indices = self.rand_idxs[idx]
        random_digits_images = self.emnist_dataset.data[rand_indices]
        transformed_random_digits_images = []
        for img in random_digits_images:
            img = tt.ToPILImage()(img)
            img = tf.resize(img,(32,32))
            img = tf.rotate(img, -90, fill=0)
            img = tf.hflip(img)
            img = tt.RandomAffine(degrees=10, translate=(0.2, 0.15), scale=(0.8, 1.1))(img)
            img = self.expand_img(img, 32, 40)
            img = tt.ToTensor()(img).numpy()
            
            transformed_random_digits_images.append(img)
        random_digits_images = np.array(transformed_random_digits_images)
        random_digits_labels = self.emnist_dataset.targets[rand_indices]
        random_sequence = np.hstack(random_digits_images.reshape((self.digits_per_sequence, 32, 40))) / 255
        random_labels = np.hstack(random_digits_labels.reshape(self.digits_per_sequence, 1))
        return random_sequence, random_labels
