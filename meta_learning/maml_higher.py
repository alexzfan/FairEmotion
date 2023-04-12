"""Implementation of model-agnostic meta-learning for Omniglot."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import wandb
import matplotlib

import affectnet_meta_util

import sys
import random
import pdb
from json import dumps

NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 2
SAVE_INTERVAL = 10
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600
RESNET_CHANNEL = 3
INNER_MODEL_SIZE = 4



class MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            num_input_channels,
            num_outputs,
            outer_lr,
            l2_wd,
            log_dir,
            weight_scheme = "global"
    ):
        """Inits MAML.

        The network consists of four convolutional blocks followed by a linear
        head layer. Each convolutional block comprises a convolution layer, a
        batch normalization layer, and ReLU activation.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.

        Args:
            num_outputs (int): dimensionality of output, i.e. number of classes
                in a task
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
                If learn_inner_lrs=True, inner_lr serves as the initialization
                of the learning rates.
            learn_inner_lrs (bool): whether to learn the above
            outer_lr (float): learning rate for outer-loop optimization
            log_dir (str): path to logging directory
        """
        self._num_input_channels = num_input_channels

        self._num_outputs = num_outputs
        # model

        self._model_ft= resnet50(pretrained = ResNet50_Weights)
        num_ftrs = self._model_ft.fc.in_features
        self._model_ft.fc = nn.Linear(num_ftrs, self._num_outputs)
        
        self._model_ft = self._model_ft.to(DEVICE)

        self._outer_lr = outer_lr
        self._optimizer = torch.optim.Adam(
            list(self._model_ft.parameters()),
            lr=self._outer_lr,
            weight_decay = l2_wd
        )

        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0
        self._weight_scheme = weight_scheme

    def _outer_step(self, task_batch, train_weights, train, step=None):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """
        loss_batch = []
        accuracy_batch = []
        for task in task_batch:
          images_full, labels_full = task
          images_full = images_full.to(DEVICE)
          labels_full = labels_full.to(DEVICE)

          if self._weight_scheme == 'global':
            logits = self._model_ft(images_full)
            loss = F.cross_entropy(logits, labels_full, weight=weights['global'], reduction = 'mean')
            loss.backward()
            _, _, acc = affectnet_meta_util.acc_score(logits, labels_full)

            loss_batch.append(loss.item())
            accuracy_batch.append(acc)
          elif self._weight_scheme == 'race_specific':
            batch_size = images_full.shape[0] // 7

            task_loss = []
            task_acc = []
            for idx, (image_batch, label_batch) in enumerate(zip(torch.split(images_full, batch_size, dim = 0), torch.split(labels_full, batch_size))):
                logits = self._model_ft(image_batch)
                loss = F.cross_entropy(logits, label_batch, weight=weights['race_specific'][idx], reduction= 'mean')
                
                # backprop the loss
                loss.backward()
                _, _, acc = affectnet_meta_util.acc_score(logits, label_batch)

                # append accs
                task_loss.append(loss.item())
                task_acc.append(acc)

            loss_batch.append(np.mean(task_loss))
            accuracy_batch.append(np.mean(task_acc))

        loss_full = np.mean(loss_batch)
        acc_full = np.mean(accuracy_batch)
        return loss_full, acc_full

    def train(self, dataloader_train, dataloader_val, writer, log):
        """Train the MAML.

        Consumes dataloader_train to optimize MAML meta-parameters
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        history_accuracy_post_adapt_query = []
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            outer_loss, accuracy_query = (
                self._outer_step(task_batch, train_weights = dataloader_train.dataset.weights, train=True, step=i_step)
            )

            self._optimizer.step()

            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {outer_loss.item():.3f}, '
                    f'accuracy: '
                    f'{accuracy_query:.3f}'
                )
                writer.add_scalar('train/loss', outer_loss.item(), i_step)
                writer.add_scalar(
                    'train/accuracy',
                    accuracy_query,
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:

                log.info(f'Evaluating at step {i_step}...')
                results, pred_dict = affectnet_meta_util.evaluate(self._model_ft, dataloader_val, DEVICE)
                
                results_str = ", ".join(f'{k}: {v:05.2f}' for k, v in results.items())
                log.info(f'Dev {results_str}')

                writer.add_scalar('val/loss', results["NLL"], i_step)
                writer.add_scalar(
                    'val/acc',
                    results["Acc"],
                    i_step
                )
                writer.add_scalar(
                    'val/f1',
                    results["F1 Score"],
                    i_step
                )
                writer.add_scalar(
                    'val/dem_parity_ratio',
                    results["Dem Parity Ratio"],
                    i_step
                )
                writer.add_scalar(
                    'val/equal_odds_ratio',
                    results["Equalized Odds Ratio"],
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

    def test(self, dataloader_test, args):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        for aug_type in ['learned','random_crop_flip','AutoAugment']:
            accuracies = []
            for task_batch in dataloader_test:
                _, _, accuracy_query = self._outer_step(task_batch, train=False, aug_type=aug_type)
                accuracies.append(accuracy_query)
            mean = np.mean(accuracies)
            std = np.std(accuracies)
            mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
            print(
                f'Augmentation type {aug_type} '
                f'Accuracy over {NUM_TEST_TASKS} test tasks: '
                f'mean {mean:.3f}, '
                f'95% confidence interval {mean_95_confidence_interval:.3f}'
            )
            wandb.log({f"{aug_type}_mean_acc": mean})
            wandb.log({f"{aug_type}_mean_95_confidence_interval":mean_95_confidence_interval})


    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._model_ft.load_state_dict(state['model_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(model_state_dict = self._model_ft.state_dict(),
                optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    # Initialize logging (Tensorboard and Wandb)
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./save/meta.batch_size:{args.batch_size}.task_batch_size:{args.task_batch_size}' # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    wandb_name = log_dir.split('/')[-1]
    if args.test : 
        wandb_name = "eval_" + wandb_name
    wandb.init(project="test-project", entity="fairemotion", config=args, name=wandb_name, sync_tensorboard=True)
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    log = affectnet_meta_util.get_logger(log_dir, "logger_name")
    # dump the args info
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    num_input_channels = 3
    maml = MAML(
      num_input_channels= num_input_channels,
      num_outputs = 7,
      outer_lr = args.outer_lr,
      l2_wd = args.l2_wd,
      log_dir = log_dir,
      weight_scheme = args.weight_scheme
    )

    if args.checkpoint_step > -1:
        log.info(f'Loading checkpoint from {args.checkpoint_step}...')
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.num_train_iterations
        log.info(f'Training for {num_training_tasks} tasks with composition: ')
            
        
        log.info("Building dataset....")
        if args.dataset == "affectnet":
            dataloader_train = affectnet_meta_util.get_affectnet_meta_dataloader(
                data_csv = args.train_csv,
                split = 'train',
                batch_size = args.batch_size,
                task_batch_size = args.task_batch_size,
                num_its_per_epoch = num_training_tasks
            )
            dataloader_val = affectnet_meta_util.get_affectnet_meta_dataloader(
                data_csv = args.val_csv,
                split = 'val',
                batch_size = args.batch_size,
                task_batch_size = None,
                num_its_per_epoch = None
            )
        else:
            raise Exception("Invalid Dataset")

        log.info("Training...")
        maml.train(
            dataloader_train,
            dataloader_val,
            writer,
            log
        )
    else:
        raise NotImplementedError("Test via CAFE")
        # print(
        #     f'Testing on tasks with composition '
        #     f'num_way={args.num_way}, '
        #     f'num_support={args.num_support}, '
        #     f'num_query={args.num_query}'
        # )
        # if args.dataset == 'omniglot':
        #     dataloader_test = omniglot.get_omniglot_dataloader(
        #         'test',
        #         1,
        #         args.num_way,
        #         args.num_support,
        #         args.num_query,
        #         NUM_TEST_TASKS
        #     )
        # elif args.dataset == 'imagenet':
        #     dataloader_test = imagenet.get_imagenet_dataloader(
        #         'test',
        #         1,
        #         args.num_way,
        #         args.num_support,
        #         args.num_query,
        #         NUM_TEST_TASKS,
        #         args.dataset_shuffle_seed
        #     )
        # elif args.dataset == "cifar":
        #     dataloader_test = cifar.get_cifar_dataloader(
        #         'test',
        #         1,
        #         args.num_way,
        #         args.num_support,
        #         args.num_query,
        #         NUM_TEST_TASKS
        #     )
        # maml.test(dataloader_test, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument("--train_csv", type = str, default="./affectnet_train_filepath_full.csv")
    parser.add_argument("--val_csv", type = str, default="./affectnet_val_filepath_full.csv")
    parser.add_argument("--dataset", type = str, default="affectnet",
                        choices = ['affectnet'])
    parser.add_argument("--weight_scheme", type = str, default="global",
                        choices = ['global', 'race_specific'])
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.0001,
                        help='outer-loop learning rate')
    parser.add_argument('--l2_wd', type=float, default=1e-4,
                        help='l2 weight decay for outer loop')            
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of images per task')
    parser.add_argument('--task_batch_size', type=int, default=4,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--debug', default=False, action = 'store_true',
                        help='debug by reducing to base maml')  
    main_args = parser.parse_args()
    main(main_args)


## Example Run command 
# python maml_higher.py --outer_lr 1e-3 --num_augs 1 --num_inner_steps 1 --aug_net_size 1 --l2_wd 1e-4 --dataset imagenet