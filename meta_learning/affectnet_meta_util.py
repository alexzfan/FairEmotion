"""Dataloading for Imagenet Tiny."""
import os
import glob
import gzip
import logging

import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import dataset, sampler, dataloader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ColorJitter, RandomHorizontalFlip 
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
import sys
from PIL import Image
from collections import OrderedDict
from sklearn import metrics
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio

NUM_TRAIN_CLASSES = 64
NUM_VAL_CLASSES = 16
NUM_TEST_CLASSES = 20
NUM_SAMPLES_PER_CLASS = 600
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(file_path):
    """Loads and transforms an imagenet tiny image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (1, 224, 224)
    """

    x = Image.open(file_path).convert("RGB")

    std_image = Compose(
            [   
                ToTensor(),
                Resize((224,224)),
                Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )
    x = std_image(x)
    return x

class AffectNetMetaDataset(dataset.Dataset):
    """affectnet dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    def __init__(self, batch_size, data_csv):
        """Inits AffectNetMetaDataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()

        # read in data
        self._data = pd.read_csv(data_csv)
        self._task_idx = {}
        for i, race in enumerate(np.unique(self._data.race)):
            self._task_idx[i] = race
        
        self._batch_size = batch_size

        # get weight info
        self.weight_dict = {}
        for idx, race in self._task_idx.items():
            temp_filter = self._data.loc[self._data['race'] == race, :]
            self.weight_dict[idx] = torch.tensor(
                compute_class_weight(
                    class_weight='balanced', 
                    classes= np.unique(temp_filter.label), 
                    y= np.array(temp_filter.label)
                    ), 
                dtype=torch.float
            ).to(DEVICE) #? should we drop the .cpu() here?

        self.total_weight = torch.tensor(
            compute_class_weight(
                class_weight='balanced', 
                classes= np.unique(self._data.label), 
                y= np.array(self._data.label)
                ), 
            dtype=torch.float
        ).to(DEVICE)

        self.weights = {
            "global": self.total_weight,
            "race_specific": self.weight_dict
        }

    def __getitem__(self, class_idxs):
        """Constructs a task.

        Data for each class is sampled uniformly at random without replacement.

        Args:
            class_idxs (tuple[int]): class indices that comprise the task

        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, channels, height, width)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,)
            images_query (Tensor): task query images
                shape (num_way * num_query, channels, height, width)
            labels_query (Tensor): task query labels
                shape (num_way * num_query,)
        """
        images_full, labels_full = [], []

        for class_idx in class_idxs:
            # get a class's examples and sample from them
            sampled_file_paths = np.random.default_rng().choice(
                self._data.loc[self._data['race'] == self._task_idx[class_idx], :],
                size=self._batch_size,
                replace=False
            )
            sample = pd.DataFrame(sampled_file_paths, columns = self._data.columns)
            images = [load_image(file_path) for file_path in sample.img_path]
            label = sample.label.tolist()
            # split sampled examples into support and query
            images_full.extend(images)
            labels_full.extend(label)

        # aggregate into tensors
        images_full = torch.stack(images_full)  # shape (7*B, C, H, W)
        labels_full = torch.tensor(labels_full)  # shape (7*B)

        return images_full, labels_full


class AffectnetRaceSampler(sampler.Sampler):
    """Samples task specification keys for an OmniglotDataset."""

    def __init__(self, num_its_per_epoch):
        """Inits OmniglotSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._split_idxs = [0,1,2,3,4,5,6]
        self._num_its_per_epoch = num_its_per_epoch

    def __iter__(self):
        return (
            self._split_idxs for _ in range(self._num_its_per_epoch)
        )

    def __len__(self):
        return self._num_its_per_epoch


def identity(x):
    return x

class AffectNetCSVDataset(dataset.Dataset):
    """
    Preprocess and prepare data for feeding into NN
    """
    def __init__(
        self,
        data_csv,
        train,
        balance = None
    ):

        # make a two col pandas df of image number : label
        self.data = pd.read_csv(data_csv)
        self.train = train

        self.label_weights = compute_class_weight(class_weight='balanced', classes= np.unique(self.data.label), y= np.array(self.data.label)) #? should we drop the .cpu() here?

    def __getitem__(self, index):

        # use the df to read in image for the given index
        image_path = self.data.loc[index, "img_path"]

        image = Image.open(image_path).convert("RGB")

        if self.train:
            std_image = Compose(
            [
                ColorJitter(brightness=0.5, hue = 0.3),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )
        else:
            std_image = Compose(
                [
                    ToTensor(),
                    Normalize(                    
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                    )
                ]
            )
        image = std_image(image)
        assert(image.shape == (3, 224, 224))

        label = self.data.loc[index, "label"]
        image_num = self.data.loc[index, 'image_num']
        example = (
            image_num,
            image,
            label
        )

        return example

    def __len__(self):

        return len(self.data)

def get_affectnet_meta_dataloader(
    data_csv,
    split,
    batch_size,
    task_batch_size,
    num_its_per_epoch
):
    """Returns a dataloader.DataLoader for affectnet meta.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """

    if split == "train":
        return dataloader.DataLoader(
            dataset=AffectNetMetaDataset(batch_size, data_csv),
            batch_size=task_batch_size,
            sampler=AffectnetRaceSampler(num_its_per_epoch),
            num_workers=8,
            collate_fn=identity,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
    elif split == "val":
        dev_dataset = AffectNetCSVDataset(data_csv, train = False, balance = False)
        return dataloader.DataLoader(dev_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8)


def acc_score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]

    preds = torch.argmax(logits, dim = -1)
    corrects_bool = preds == labels
    corrects_bool = corrects_bool.type(torch.float)

    num_correct = torch.sum(corrects_bool).item()
    acc = torch.mean(corrects_bool).item()
    return preds, num_correct, acc

class AverageMeter:
    """Keep track of average values over time.
    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.
        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count

def evaluate(model, data_loader, device):
    nll_meter = AverageMeter()

    model.eval()
    pred_dict = {} # id, prob and prediction
    full_labels = []
    predictions = []
    race_labs = data_loader.dataset.data['race']

    acc = 0
    num_corrects, num_samples = 0, 0

    with torch.no_grad():
        for img_id, x, y in data_loader:
            # forward pass here
            x = x.float().to(device)

            batch_size = x.shape[0]

            score = model(x)

            # calc loss
            y = y.type(torch.LongTensor).to(device)
            criterion = nn.CrossEntropyLoss()

            preds, num_correct, acc = acc_score(score, y) 
            loss = criterion(score, y)

            loss_val = loss.item() 
            nll_meter.update(loss_val, batch_size)

            # get acc and auroc
            num_corrects += num_correct
            num_samples += preds.size(0)
            predictions.extend(preds)
            full_labels.extend(y)


        acc = float(num_corrects) / num_samples

        # F1 Score
        y_pred = np.asarray([pred.cpu() for pred in predictions]).astype(int)
        y = np.asarray([label.cpu() for label in full_labels]).astype(int)
        f1 = metrics.f1_score(y, y_pred, average = 'macro')

        # fairness metrics
        dem_parity_ratio = demographic_parity_ratio(y_true = y, 
                                                    y_pred = y_pred, 
                                                    sensitive_features = race_labs)
        equal_odds_ratio = equalized_odds_ratio(y_true = y, 
                                                y_pred = y_pred, 
                                                sensitive_features = race_labs)
        
    model.train()

    results_list = [("NLL", nll_meter.avg),
                    ("Acc", acc),
                    ("F1 Score", f1),
                    ("Dem Parity Ratio", dem_parity_ratio),
                    ("Equalized Odds Ratio", equal_odds_ratio),]
    results = OrderedDict(results_list)
    return results, pred_dict

def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.
    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.
    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.
        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger