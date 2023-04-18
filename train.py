"""
Train a model on the AffectNet Dataset
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import util
import wandb

from args import get_train_args
from models import baseline_pretrain
from util import AffectNetDataset, AffectNetCSVDataset, CAFEDataset
from collections import OrderedDict
from sklearn import metrics
from tensorboardX import SummaryWriter
from tqdm import tqdm
from json import dumps
from sklearn.utils.class_weight import compute_class_weight
import os
import sys

def main(args):

    # set up logger and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training = True)
    log = util.get_logger(args.save_dir, args.name)
    

    wandb_name = args.save_dir.split('/')[-1]
    wandb.init(project = 'test-project', entity = 'fairemotion', config = args, name = wandb_name, sync_tensorboard = True)
    tbx = SummaryWriter(args.save_dir)

    device, args.gpu_ids = util.get_available_devices()

    # dump the args info
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # set seed
    log.info(f'Seed: {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get Model
    log.info("Making model....")
    if(args.model_type == "baseline"):
        model = baseline_pretrain(7)
    else:
        raise Exception("Model provided not valid")

    model = nn.DataParallel(model, args.gpu_ids)

    # load the step if restarting
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        if args.load_path_step_bool:
            model, step = util.load_model(model, args.load_path, args.gpu_ids)
        else:
            model = util.load_model(model, args.load_path, args.gpu_ids, return_step = False)
            step = 0
    else:
        step = 0

    # send model to dev and start training
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get checkpoint saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints = args.max_checkpoints,
                                 metric_name = args.metric_name,
                                 maximize_metric = args.maximize_metric,
                                 log = log)
    optimizer = optim.Adam(model.parameters(),
                            lr = args.lr,
                            betas = (0.9, 0.999),
                            eps = 1e-7,
                            weight_decay = args.l2_wd)


    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # load in data
    log.info("Building dataset....")
    if(args.dataset == "affectnet"):
        train_dataset = AffectNetCSVDataset(args.train_csv, train = False, balance = False)
        train_loader = data.DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)
        dev_dataset = AffectNetCSVDataset(args.val_csv, train = False, balance = False)
        dev_loader = data.DataLoader(dev_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers)
    elif(args.dataset == "cafe"):
        raise Exception("do not finetune on cafe")
        # train_dataset = CAFEDataset(args.cafe_train_csv, train = True, balance = False)
        # train_loader = data.DataLoader(train_dataset,
        #                             batch_size=args.batch_size,
        #                             shuffle=True,
        #                             num_workers=args.num_workers)
        # dev_dataset = CAFEDataset(args.cafe_val_csv, train = False, balance = False)
        # dev_loader = data.DataLoader(dev_dataset,
        #                             batch_size=args.batch_size,
        #                             shuffle=False,
        #                             num_workers=args.num_workers)
    else:
        raise Exception("Dataset provided not valid")
    # Start training
    log.info("Training...")
    steps_till_eval = args.eval_steps

    # calc epoch 
    epoch = step // len(train_dataset)
    

    weights = torch.tensor(train_dataset.label_weights,dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction= 'mean')

    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}....')
        with torch.enable_grad(), \
            tqdm(total=len(train_loader.dataset)) as progress_bar:
            for img_id, x, y, race in train_loader:
                # forward pass here
                x = x.float().to(device)

                batch_size = args.batch_size
                optimizer.zero_grad()

                if(args.model_type == "baseline"):
                    score = model(x)
                else:
                    raise Exception("Model Type is Invalid")

                # calc loss
                y = y.type(torch.LongTensor).to(device)
                loss = criterion(score, y) # do not need unsqueeze for CCELoss I think?
                loss_val = loss.item()

                # backward pass here
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step//batch_size)
                ema(model, step//batch_size)

                # log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch = epoch,
                                         loss = loss_val)
                tbx.add_scalar("train/loss", loss_val, step)
                tbx.add_scalar("train/LR", optimizer.param_groups[0]['lr'],
                               step)
                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Eval and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(args,
                                                  model,
                                                  dev_loader,
                                                  device)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    results_str = ", ".join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # tensorboard
                    log.info("Visualizing in TensorBoard")
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)

def evaluate(args, model, data_loader, device):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {} # id, prob and prediction
    full_labels = []
    predictions = []
    race_labs = data_loader.dataset.data['race']
    test = []

    acc = 0
    num_corrects, num_samples = 0, 0

    with torch.no_grad(), \
        tqdm(total=len(data_loader.dataset)) as progress_bar:
        for img_id, x, y, race in data_loader:
            # forward pass here
            x = x.float().to(device)

            batch_size = args.batch_size

            if(args.model_type == "baseline"):
                score = model(x)
            else:
                raise Exception("Model Type Invalid")

            # calc loss
            y = y.type(torch.LongTensor).to(device)
            criterion = nn.CrossEntropyLoss()

            preds, num_correct, acc = util.acc_score(score, y) #? should we unsqueeze
            loss = criterion(score, y)

            loss_val = loss.item() # ? same here
            nll_meter.update(loss_val, batch_size)

            # get acc and auroc
            num_corrects += num_correct
            num_samples += preds.size(0)
            predictions.extend(preds)
            full_labels.extend(y)
            test.extend(race)



        acc = float(num_corrects) / num_samples

        # F1 Score
        y_pred = np.asarray([pred.cpu() for pred in predictions]).astype(int)
        y = np.asarray([label.cpu() for label in full_labels]).astype(int)
        f1 = metrics.f1_score(y, y_pred, average = 'weighted')

        # dem parity ratio
        fairlearn_dem_parity_ratio = demographic_parity_ratio(y_true = y, 
                                                    y_pred = y_pred, 
                                                    sensitive_features = race_labs)

        # weighted OVO fairness metrics
        truth_sample_size_weights = pd.Series(y).value_counts(normalize = True).sort_index().tolist()
        assert(len(truth_sample_size_weights) == 7)

        dem_parity_ratio = 0
        eq_odds_ratio = 0
        for i, weight in enumerate(truth_sample_size_weights):
            y_true_converted = [1 if j == i else 0 for j in y]
            y_pred_converted = [1 if j == i else 0 for j in y_pred]
            dem_parity_ratio += weight*demographic_parity_ratio(y_true = y_true_converted, 
                                            y_pred = y_pred_converted,
                                            sensitive_features = race_labs
                                            )
            eq_odds_ratio += weight*equalized_odds_ratio(y_true = y_true_converted,
                                                        y_pred = y_pred_converted,
                                                        sensitive_features = race_labs,
                                                        )
        

    model.train()

    results_list = [("NLL", nll_meter.avg),
                    ("Acc", acc),
                    ("F1 Score", f1),
                    ("FL Dem Parity Ratio", fairlearn_dem_parity_ratio),
                    ("OVO Dem Parity Ratio", dem_parity_ratio),
                    ("OVO Eq Odds Ratio", eq_odds_ratio),]
    results = OrderedDict(results_list)

    return results, pred_dict




if __name__ == '__main__':
    main(get_train_args())