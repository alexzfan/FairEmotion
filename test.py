"""
Test a model on the CAFE
"""
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_test_args
from models import baseline_pretrain
from util import AffectNetCSVDataset, CAFEDataset
from collections import OrderedDict
from sklearn import metrics
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from json import dumps
import os
from os.path import join

def main(args):
    # Set up logging
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # set up logger and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training = False)
    log = util.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = util.get_available_devices()

    # dump the args info
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Get Model
    log.info("Making model....")
    if(args.model_type == "baseline"):
        model = baseline_pretrain(7)
    else:
        raise Exception("Model provided not valid")

    model = nn.DataParallel(model, args.gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')

    # get data loader
    if(args.dataset == "cafe"):
        test_csv = args.cafe_test_csv
        test_dataset = CAFEDataset(test_csv, train = False, balance = False)
        test_loader = data.DataLoader(test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers)
    elif(args.dataset == 'affectnet'):
        # mainly for race bias quantifying
        test_csv = args.affectnet_test_csv
        test_dataset = AffectNetCSVDataset(test_csv, 
                                            train = False, 
                                            balance = False)
        test_loader = data.DataLoader(test_dataset,
                                    batch_size = args.batch_size,
                                    shuffle = False,
                                    num_workers = args.num_workers)
    else:
        raise Exception("Dataset provided not valid")

    log.info(f"Evaluating on {test_csv} split...")
    nll_meter = util.AverageMeter()

    model = util.load_model(model, args.load_path, args.gpu_ids, return_step = False)
    model = model.to(device)
    model.eval()

    full_labels = []
    full_img_id = []
    full_preds = []

    acc = 0
    num_corrects, num_samples = 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad(), \
        tqdm(total=len(test_dataset)) as progress_bar:
        for img_id, x, y, _ in test_loader:
            # forward pass here
            x = x.float().to(device)
            # text = text.to(device)

            batch_size = args.batch_size

            if(args.model_type == "baseline"):
                score = model(x)
            else:
                raise Exception("Model Type Invalid")

            # calc loss
            y = y.type(torch.LongTensor).to(device)
            preds, num_correct, acc = util.acc_score(score, y)

            loss = criterion(score, y)
            nll_meter.update(loss.item(), batch_size)
            progress_bar.update(batch_size)
            progress_bar.set_postfix(loss = loss.item())
             # get acc and auroc
            num_corrects += num_correct
            num_samples += preds.size(0)

            full_img_id.extend([ids.item() for ids in img_id])
            full_preds.extend(preds)
            full_labels.extend(y)

        acc = float(num_corrects) / num_samples

        # F1 Score
        y_pred = np.asarray([pred.cpu() for pred in full_preds]).astype(int)
        y = np.asarray([label.cpu() for label in full_labels]).astype(int)
        f1 = metrics.f1_score(y, y_pred, average = 'macro')

        df = pd.DataFrame(list(zip(full_img_id, y_pred, y)), columns =['img_id', 'preds', 'labels'])
        sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
        df.to_csv(sub_path, encoding = "utf-8")

        print("Acc: {}, F1: {}".format(acc, f1))



if __name__ == '__main__':
    main(get_test_args())
