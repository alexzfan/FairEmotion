"""
Utility classes and methods for long covid classification
"""


from copy import deepcopy
import glob
import logging
import os
import queue
from PIL import Image
import re
import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.utils.class_weight import compute_class_weight


class AffectNetDataset(data.Dataset):
    """
    Preprocess and prepare data for feeding into NN
    """
    def __init__(
        self,
        data_dir,
        train,
        balance = None
    ):

        # make a two col pandas df of image number : label
        self.data_dir = data_dir
        self.train = train
        annotations = glob.glob(os.path.join(data_dir, "annotations", "*_exp.npy"))

        image_nums = []
        labels = []
        for annotation in annotations:
            image_num = re.findall(r'(\d+)', str(annotation))[0]
            label = np.load(annotation).item()

            if label == '7':
                continue

            image_nums.append(image_num)
            labels.append(label)
        
        self.data = pd.DataFrame({"image_num": image_nums, "label": labels})
        self.data.label = self.data.label.astype(int)

        if balance: # !ignore for now, not downsampling
        

            # downsample here
            g = data.groupby(label, group_keys=False)
            self.data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()))).reset_index(drop=True)

        self.label_weights = compute_class_weight(class_weight='balanced', classes= np.unique(labels), y= np.array(labels)) #? should we drop the .cpu() here?

    def __getitem__(self, index):

        # use the df to read in image for the given index
        image_path = os.path.join(self.data_dir, "images", self.data.loc[index, "image_num"] + ".jpg")

        image = Image.open(image_path).convert("RGB")

        if self.train:
            std_image = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5, hue = 0.3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )
        else:
            std_image = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(                    
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

class AffectNetCSVDataset(data.Dataset):
    """
    Preprocess and prepare data for feeding into NN
    """
    def __init__(
        self,
        data_csv,
        train,
        balance = None,
        race_quant_sampling = None,
        race_quant_sampling_prop = None
    ):

        # make a two col pandas df of image number : label
        self.data = pd.read_csv(data_csv)
        self.train = train

        self.label_weights = compute_class_weight(class_weight='balanced', classes= np.unique(self.data.label), y= np.array(self.data.label)) #? should we drop the .cpu() here?

        if race_quant_sampling and race_quant_sampling_prop:
            assert(race_quant_sampling in np.unique(self.data.race))
            assert(isinstance(race_quant_sampling_prop, int))

            if race_quant_sampling == "White":
                # down sample white, keep all else
                non_white = self.data[self.data.race != "White"]
                num_non_white = len(non_white)

                num_white = np.floor((race_quant_sampling_prop*num_non_white) / (1-race_quant_sampling_prop)).astype(int)
                white = self.data[self.data.race == "White"].sample(num_white)
                self.data = pd.concat([white, non_white]).reset_index(drop = True)

                assert(np.isclose(num_white/len(self.data), race_quant_sampling_prop, atol = 1e-2))
                
            else:
                spec_race = self.data[self.data.race == race_quant_sampling]
                num_spec_race = len(spec_race)

                num_others = np.floor((num_spec_race*(1-race_quant_sampling_prop))/race_quant_sampling_prop).astype(int)
                others = self.data[self.data.race != race_quant_sampling].sample(num_others)
                self.data = pd.concat([spec_race, others]).reset_index(drop = True)

                assert(np.isclose(num_spec_race/len(self.data), race_quant_sampling_prop, atol = 1e-2))


    def __getitem__(self, index):

        # use the df to read in image for the given index
        image_path = self.data.loc[index, "img_path"]

        image = Image.open(image_path).convert("RGB")

        if self.train:
            std_image = transforms.Compose(
            [
                # transforms.ColorJitter(brightness=0.5, hue = 0.3),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )
        else:
            std_image = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(                    
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                    )
                ]
            )
        image = std_image(image)
        assert(image.shape == (3, 224, 224))

        label = self.data.loc[index, "label"]
        image_num = self.data.loc[index, 'image_num']
        race = self.data.loc[index, 'race']
        example = (
            image_num,
            image,
            label,
            race
        )

        return example

    def __len__(self):

        return len(self.data)

class CAFEDataset(data.Dataset):
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
        assert(os.path.isdir("./cropped_sessions"))
        self.data_csv = data_csv
        self.train = train
        
        self.data = pd.read_csv(data_csv)
        self.data.label = self.data.label.astype(int)
        self.labels = self.data.label.tolist()

        self.label_weights = compute_class_weight(class_weight='balanced', classes= np.unique(self.labels), y= np.array(self.labels)) #? should we drop the .cpu() here?

    def __getitem__(self, index):

        # use the df to read in image for the given index
        image_path = "cropped_" + self.data.loc[index, 'Orig_filepath']

        image = Image.open(image_path).convert("RGB")

        if self.train:
            std_image = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5, hue = 0.3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )
        else:
            std_image = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224,224)),
                    transforms.Normalize(                    
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                    )
                ]
            )
        image = std_image(image)
        if(image.shape != (3, 224, 224)):
            raise Exception("image shape wrong. {}, {}".format(self.data.loc[index,'Internal_id'], image.shape))

        label = self.data.loc[index, "label"]
        internal_id = self.data.loc[index, 'Internal_id']

        example = (
            internal_id,
            image,
            label
        )

        return example

    def __len__(self):

        return len(self.data)

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

class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.
        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass

class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

def make_update_dict(img_ids, preds, scores, labels):
    pred_dict = {}

    for img_id, pred, score, label in zip(img_ids, preds, scores, labels):
        pred_dict[img_id] = (score, pred, label)

    return pred_dict


def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')

def get_available_devices():
    """Get IDs of all available GPUs.
    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
        print('using GPU')
    else:
        device = torch.device('cpu')
        print('using cpu')

    return device, gpu_ids


def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.
    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.
    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model

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

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return y_pred_tag, correct_results_sum, acc



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

# def collate_fn(batch):
#     r"""Puts each data field into a tensor with outer dimension batch size"""

#     error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
#     elem_type = type(batch[0])
#     if isinstance(batch[0], torch.Tensor):
#         out = None
#         if _use_shared_memory:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum([x.numel() for x in batch])
#             storage = batch[0].storage()._new_shared(numel)
#             out = batch[0].new(storage)
#         return torch.stack(batch, 0, out=out)
#     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#             and elem_type.__name__ != 'string_':
#         elem = batch[0]
#         if elem_type.__name__ == 'ndarray':
#             # array of string classes and object
#             if re.search('[SaUO]', elem.dtype.str) is not None:
#                 raise TypeError(error_msg.format(elem.dtype))

#             return torch.stack([torch.from_numpy(b) for b in batch], 0)
#         if elem.shape == ():  # scalars
#             py_type = float if elem.dtype.name.startswith('float') else int
#             return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
#     elif isinstance(batch[0], int_classes):
#         return torch.LongTensor(batch)
#     elif isinstance(batch[0], float):
#         return torch.DoubleTensor(batch)
#     elif isinstance(batch[0], string_classes):
#         return batch
#     elif isinstance(batch[0], container_abcs.Mapping):
#         return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
#     elif isinstance(batch[0], container_abcs.Sequence):
#         transposed = zip(*batch)
#         return [default_collate(samples) for samples in transposed]

#     raise TypeError((error_msg.format(type(batch[0]))))