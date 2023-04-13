import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class AdversarialDataset(Dataset):
    def __init__(self, batch_size, data_file, transform = None):
        super().__init__()
        # read data file
        self._data = data_file
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
