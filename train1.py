import logging

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import importlib
import datetime
from config import model_name
from dataset import BaseDataset
from evaluate import evaluate

logger = logging.getLogger('AutoML')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    Config = getattr(importlib.import_module('config'), f"{model_name}Config")
except (AttributeError, ModuleNotFoundError):
    print(f"{model_name} not included!")
    exit()


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def train():
    writer = None

    batchs = Config.batch_size

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load('data/train_l/pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    model = Model(Config, pretrained_word_embedding).to(device)

    print(model)

    dataset = BaseDataset('data/train_l/behaviors_parsed.tsv',
                          'data/train_l/news_parsed.tsv',
                          Config.dataset_attributes)

    print(f"Load training dataset with size {len(dataset)}.")

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=batchs,
                   shuffle=True,
                   num_workers=Config.num_workers,
                   drop_last=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping()

    checkpoint_dir = os.path.join('./checkpoint', model_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if Config.load_checkpoint:
        checkpoint_path = latest_checkpoint(checkpoint_dir)
        if checkpoint_path is not None:
            print(f"Load saved parameters in {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            step = checkpoint['step']
            if 'early_stop_value' in checkpoint:
                early_stopping(checkpoint['early_stop_value'])
            model.train()

    with tqdm(total=Config.num_batches, desc="Training") as pbar:
        for i in range(1, Config.num_batches + 1):
            try:
                minibatch = next(dataloader)
            except StopIteration:
                exhaustion_count += 1
                tqdm.write(
                    f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
                )
                dataloader = iter(
                    DataLoader(dataset,
                               batch_size=batchs,
                               shuffle=True,
                               num_workers=Config.num_workers,
                               drop_last=True))
                minibatch = next(dataloader)

            step += 1

            y_pred = model(minibatch['clicked_news_length'],
                           minibatch["candidate_news"], minibatch["clicked_news"])

            loss = torch.stack([x[0] for x in -F.log_softmax(y_pred, dim=1)
                                ]).mean()

            loss_full.append(loss.item())
            loss.backward()

            if ((i + 1) % Config.accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % 10 == 0:
                pass

            if i % Config.num_batches_show_loss == 0:
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}"
                )

            if i % Config.num_batches_validate == 0:
                val_auc, val_mrr, val_ndcg, val_ndcg5, val_ndcg10 = evaluate(
                    model, 'data/val_l')

                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f},"
                    f"validation nDCG: {val_ndcg:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
                )
                early_stop, get_better = early_stopping(-val_auc)
                if early_stop:
                    logger.debug('Final result is %g', val_auc)
                    logger.debug('Send final result done.')
                    tqdm.write('Early stop.')
                    break
                elif get_better:
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': step,
                            'early_stop_value': -val_auc
                        }, f"./checkpoint/{model_name}/ckpt-{step}.pth")

            pbar.update(1)


def time_since(since):
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':

    print('Using device:', device)
    print(f'Training model {model_name}')
    '''@nni.get_next_parameter()'''
    try:
        train()
    except Exception as exception:
        logger.exception(exception)
        print("Training error...")
        raise
