import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ColorJitter, RandomHorizontalFlip
from PIL import Image

TRANSFORM = Compose([
                ToTensor(),
                Resize((224,224)),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )

class AdversarialDataset(Dataset):
    def __init__(self, batch_size, data_file, transform = TRANSFORM):
        super().__init__()
        # read data file
        self._data = pd.read_csv(data_file)
        self._task_idx = dict()
        for i, race in enumerate(np.unique(self._data.race)):
            self._task_idx[i] = race

        self._batch_size = batch_size
        self.transform = transform

    def __getitem__(self, class_indices):
        images, labels  = [], []
        for class_idx in class_indices:
            sampled_file_paths = np.random.default_rng().choice(
                self._data.loc[self._data['race'] == self._task_idx[class_idx], :],
                size=self._batch_size,
                replace=False
            )
            sample = pd.DataFrame(sampled_file_paths, columns = self._data.columns)
            images = [Image.open(file_path).convert("RGB") for file_path in sample.img_path]
            if self.transform is not None:
                images = [self.transform(img) for img in images]
            label = sample.label.tolist()
            # split sampled examples into support and query
            images.extend(images)
            labels.extend(label)

        return images, labels

def get_adversary_dataloader(data_csv, split, batch_size, task_batch_size):
    if split == "train":
        dataset = AdversarialDataset(batch_size, data_csv)
        return DataLoader(dataset, batch_size = task_batch_size)

    elif split == "val":
        dataset = AdversarialDataset(batch_size, data_csv)
        return DataLoader(dataset, batch_size = task_batch_size, shuffle = False)

    elif split == "test":
        dataset = AdversarialDataset(batch_size, data_csv)
        return DataLoader(dataset, batch_size = task_batch_size, shuffle = False)
