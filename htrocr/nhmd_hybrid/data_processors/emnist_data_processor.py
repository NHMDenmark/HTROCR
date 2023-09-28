import htrocr.nhmd_hybrid.utils.device as device
from torchvision import datasets
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as tt
import torchvision.transforms.functional as tf
import torch

def prepare_data(usedeviceload=True):
    digits_per_sequence = 5
    number_of_sequences = 5000
    emnist_dataset = datasets.EMNIST('./EMNIST', split="digits", train=True, download=True)
    dataset_sequences = []
    dataset_labels = []
    dataset_origw = []

    for i in range(number_of_sequences):
        random_indices = np.random.randint(len(emnist_dataset.data), size=(digits_per_sequence,))
        random_digits_images = emnist_dataset.data[random_indices]
        transformed_random_digits_images = []

        for img in random_digits_images:
            img = tt.ToPILImage()(img)
#            img = tf.resize(img,(40,40))
            img = tf.rotate(img, -90, fill=0)
            img = tf.hflip(img)
            img = tt.RandomAffine(degrees=10, translate=(0.2, 0.15), scale=(0.8, 1.1))(img)
            img = tt.ToTensor()(img).numpy()
            transformed_random_digits_images.append(img)

        random_digits_images = np.array(transformed_random_digits_images)
        random_digits_labels = emnist_dataset.targets[random_indices]
#        random_sequence = np.hstack(random_digits_images.reshape((digits_per_sequence, 40, 40)))
        random_sequence = np.hstack(random_digits_images.reshape((digits_per_sequence, 28, 28)))
        random_labels = np.hstack(random_digits_labels.reshape(digits_per_sequence, 1))
        dataset_sequences.append(random_sequence / 255)
        dataset_origw.append(200)
        dataset_labels.append(random_labels)

    dataset_data = torch.Tensor(np.array(dataset_sequences))
    dataset_labels = torch.IntTensor(np.array(dataset_labels))
    dataset_origw = torch.IntTensor(np.array(dataset_origw))

    seq_dataset = TensorDataset(dataset_data, dataset_labels, dataset_origw)
    train_set, val_set = torch.utils.data.random_split(seq_dataset,
                                                    [int(len(seq_dataset) * 0.8), int(len(seq_dataset) * 0.2)])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)
    dev = device.default_device()
    if usedeviceload: 
        train_dl = device.DeviceDataLoader(train_loader, dev)
        valid_dl = device.DeviceDataLoader(val_loader, dev)
    else:
        train_dl = train_loader
        valid_dl = val_loader
    return train_dl, valid_dl 
