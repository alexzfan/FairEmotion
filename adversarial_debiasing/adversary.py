import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from torch.utils import tensorboard
import wandb

from adversarial_dataset import get_adversary_dataloader, evaluate, get_logger

import argparse
import os
import sys
import random
import pdb
from tqdm import tqdm
from json import dumps

import nvidia_smi
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class baseline_classifier(nn.Module):
    def __init__(self, num_classes = 7):
        super(baseline_classifier, self).__init__()
        self.model_ft= resnet50(pretrained = ResNet50_Weights)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, num_classes)

        self.model_ft = self.model_ft.to(DEVICE)

    def forward(self, x):
        # forward through linear layers
        out = self.model_ft(x)
        return out

class adversary_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_races = 7):
        super(adversary_classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_races)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # returns the adversary logits
        return x

def save(classifier, adversary,
        optimizer_cls, optimizer_adv,
    log_dir, checkpoint_step):
    target_path = (
        f'{os.path.join(log_dir, "state")}'
        f'{checkpoint_step}.pt'
    )

    optimizer_cls_state_dict = optimizer_cls.state_dict()
    optimizer_adv_state_dict = optimizer_adv.state_dict()

    classifier_state_dict = classifier.state_dict()
    adversary_state_dict = adversary.state_dict()
    torch.save(
        dict(
            classifier_state_dict = classifier_state_dict,
            adversary_state_dict = adversary_state_dict, 
            optimizer_cls_state_dict = optimizer_cls_state_dict,
            optimizer_adv_state_dict = optimizer_adv_state_dict
        ),
        target_path
    )
    return

def load(classifier, adversary,
        optimizer_cls, optimizer_adv,
    log_dir, checkpoint_step):

    target_path = (
        f'{os.path.join(log_dir, "state")}'
        f'{checkpoint_step}.pt'
    )

    if os.path.isfile(target_path):
        state = torch.load(target_path)
        classifier.load_state_dict(state['classifier_state_dict'])
        adversary.load_state_dict(state['adversary_state_dict'])
        optimizer_cls.load_state_dict(state['optimizer_cls_state_dict'])
        optimizer_adv.load_state_dict(state['optimizer_adv_state_dict'])
        print(f'Loaded checkpoint iteration {checkpoint_step}.')
    else:
        raise ValueError(
            f'No checkpoint for iteration {checkpoint_step} found.'
        )
    return classifier, adversary, optimizer_cls, optimizer_adv
    

def classifier_train(classifier, adversary,
                    train_loader, val_loader,
                    optimizer_cls, optimizer_adv,
                    adv_alpha, 
                    num_epochs,
                    step, eval_step,
                    writer, log_dir, # tensorboard
                    log,
                    device = DEVICE):

    steps_till_eval = eval_step

    epoch = step // len(train_loader)

    weights = torch.tensor(train_loader.dataset.label_weights, dtype = torch.float).to(device)

    while epoch != num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}')

        classifier.train()
        adversary.train()

        with tqdm(total=len(train_loader.dataset)) as progress_bar:
            for batch_idx, (image_num, data, label, group) in enumerate(train_loader):
                data, label, group = data.to(device), label.to(device), group.to(device)
                batch_size = data.shape[0]

                # zero out optimizers
                optimizer_cls.zero_grad()
                optimizer_adv.zero_grad()

                # first predict emotion labels
                preds = classifier(data)
                loss_cls = F.cross_entropy(preds, label, weight = weights)
                pred_loss_val = loss_cls.item()

                # get dW_LP
                loss_cls.backward(retain_graph = True)
                dW_LP = [torch.clone(p.grad.detach()) for p in classifier.parameters()]
                # dW_LP = autograd.grad(
                #     outputs = loss_cls,
                #     inputs = classifier.parameters(),
                #     retain_graph = True
                # )
                optimizer_cls.zero_grad()
                optimizer_adv.zero_grad()

                # predict the adversary's race labels
                output_adv = adversary(preds)
                loss_adv = F.cross_entropy(output_adv, group)
                adv_loss_val = loss_adv.item()

                # backward and obtain dW_LA
                loss_adv.backward(retain_graph=False)
                dW_LA = [torch.clone(p.grad.detach()) for p in classifier.parameters()]

                for i, param in enumerate(classifier.parameters()):
                    # normalize dW_LA
                    unit_dW_LA = dW_LA[i] / (torch.norm(dW_LA[i]) +torch.finfo(float).tiny)
                    # draw projection
                    proj = torch.sum(torch.mul(unit_dW_LA, dW_LP[i]))
                    # compute dW
                    param.grad = dW_LP[i] - (proj*unit_dW_LA) - (adv_alpha*dW_LA[i])
                    
                optimizer_cls.step()
                optimizer_adv.step()

                # log train info
                step += batch_size
                writer.add_scalar("train/pred_loss",pred_loss_val, step)
                writer.add_scalar("train/adv_loss", adv_loss_val, step)
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch = epoch,
                                         loss = pred_loss_val)
                steps_till_eval -= batch_size

                if steps_till_eval <= 0:
                    steps_till_eval = eval_step
                    
                    # evaluate using just predictor net
                    results, pred_dict = evaluate(classifier, val_loader, device)
                    results_str = ", ".join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Val {results_str}')
                
                    for k, v in results.items():
                        writer.add_scalar(f'val/{k}', v, step)

                    # save validation step
                    save(
                        classifier,
                        adversary,
                        optimizer_cls,
                        optimizer_adv,
                        log_dir,
                        step
                    )

def test(classifier, adversary, test_loader, device = DEVICE):
    classifier.eval()

    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for (img, label, group) in test_loader:
            data, label, group = data.to(device), label.to(device), group.to(device)
            output = classifier(img)
            adv_output = adversary(output)
            adv_pred = torch.argmax(adv_output, dim=1)
            debiased_pred = torch.zeros_like(adv_pred)

            for i in range(len(debiased_pred)):
                debiased_pred[i] = output[i, adv_pred[i]]

            debiased_probs = torch.nn.functional.softmax(debiased_pred, dim=1)
            debiased_labels = torch.argmax(debiased_probs, dim=1)

            test_loss += F.cross_entropy(debiased_labels, label)
            test_acc += (debiased_labels == label).float().mean()

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    print(f"Test Loss={test_loss.item():.4f}, Test Accuracy={test_acc.item()*100:.2f}%")

def main(args):

    # Initialize logging (Tensorboard and Wandb)
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./save/adversarial.batch_size:{args.batch_size}' # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    wandb_name = log_dir.split('/')[-1]
    wandb.init(project="test-project", entity="fairemotion", config=args, name=wandb_name, sync_tensorboard=True)
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    log = get_logger(log_dir, "logger_name")
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    # make the classifiers
    predictor = baseline_classifier(num_classes = 7)
    adversary = adversary_classifier(input_size = 7, 
                                    hidden_size = args.adv_hidden_size, 
                                    num_races =7)

    # Define the optimizer and hyparameters for training the adversarial classifier
    optimizer_predictor = optim.Adam(predictor.parameters(), lr = args.predictor_lr)
    optimizer_adv = optim.Adam(adversary.parameters(), lr = args.adversary_lr)
    step = 0
    
    # load in if checkpoint step defined
    if args.checkpoint_step > -1: 
        predictor, adversary, optimizer_predictor, optimizer_adv = load(
            predictor,
            adversary,
            optimizer_predictor,
            optimizer_adv,
            log_dir,
            checkpoint_step
        )
        # update starting step
        step = checkpoint_step

    predictor = predictor.to(DEVICE)
    adversary = adversary.to(DEVICE)

    # Define dataloaders
    if args.test:
        raise Exception("Test not implemented yet")
    else:
        train_loader = get_adversary_dataloader(data_csv = args.train_csv,
                    split = 'train',
                    batch_size = args.batch_size)

        val_loader = get_adversary_dataloader(data_csv = args.val_csv,
                    split = 'val',
                    batch_size = args.batch_size)

        classifier_train(classifier = predictor, 
                        adversary = adversary,
                        train_loader = train_loader, 
                        val_loader = val_loader,
                        optimizer_cls =optimizer_predictor,
                        optimizer_adv = optimizer_adv,
                        adv_alpha = args.adversary_alpha, 
                        num_epochs = args.num_epochs,
                        step = step, 
                        eval_step = args.eval_step,
                        writer = writer, 
                        log_dir = log_dir,
                        log = log)


if __name__=='__main__':
    parser = argparse.ArgumentParser("Train Adversarial Debiasing on Facial Expression")

    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument("--train_csv", type = str, default="./affectnet_train_filepath_full.csv")
    parser.add_argument("--val_csv", type = str, default="./affectnet_val_filepath_full.csv")
    parser.add_argument('--predictor_lr', type=float, default=0.0001,
                        help='predictor learning rate')
    parser.add_argument('--adversary_lr', type=float, default=0.001,
                        help='adversary learning rate')
    parser.add_argument('--adversary_alpha', type=float, default=1,
                        help='adversary alpha')
    parser.add_argument('--dynamic_alpha', default=False, action = 'store_true',
                        help='adjust alpha as 1/sqrt(t) and predictor_lr as 1/t')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of images per batch')
    parser.add_argument('--adv_hidden_size', type=int, default=64,
                        help='hidden size of adv layers')
    parser.add_argument('--l2_wd', type=float, default=0,
                        help='l2 weight decay for outer loop')
    parser.add_argument('--eval_step', type=int, default=40000,
                        help='number of steps before eval for validation')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--test', default=False, action = 'store_true',
                        help='Test on CAFE')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number epochs')    
    main_args = parser.parse_args()
    main(main_args)     