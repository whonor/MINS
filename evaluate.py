import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from config import model_name
from torch.utils.data import Dataset, DataLoader
import os
import json
import pandas as pd
from ast import literal_eval
import importlib

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    Config = getattr(importlib.import_module('config'), f"{model_name}Config")
except (AttributeError, ModuleNotFoundError):
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dcg_score(y_true, y_score, k=None):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=None):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def mrrAtK(pScore, nScore, atK=None):
    """
    MRR@
    pScore: scores for positive instances
    nScore: scores for negative instances
    atK:    top K
    """
    pScore = np.asarray(pScore).flatten()
    nScore = np.asarray(nScore).flatten()

    T = len(pScore)

    if atK is None:
        atK = T

    mrr = np.zeros_like(atK, dtype=float)
    for p in pScore:
        rank = np.sum(nScore > p) + 1
        mrr += (rank <= atK) * (1 / rank)

    return mrr / T


def precisionAtK(pScore, nScore, atK=None):
    """
    Precision@
    pScore: scores for positive instances
    nScore: scores for negative instances
    atK:    top K
    """
    pScore = np.asarray(pScore).flatten()
    nScore = np.asarray(nScore).flatten()
    T = len(pScore)
    N = len(nScore)
    R = T + N

    if atK is None: atK = T

    IX = (-np.concatenate([pScore, nScore])).argsort()
    rankvec = np.concatenate([np.ones(T), np.zeros(N)])
    rankvec = np.cumsum(rankvec[IX])

    topK = np.minimum(atK, R)
    hits = rankvec[topK - 1]
    return hits / topK


def avgPrecisionAtK(pScore, nScore, atK=None):
    """
    MAP@
    pScore: scores for positive instances
    nScore: scores for negative instances
    atK:    top K
    """
    pScore = np.asarray(pScore).flatten()
    nScore = np.asarray(nScore).flatten()
    T = len(pScore)
    N = len(nScore)
    R = T + N

    if atK is None: atK = T

    IX = (-np.concatenate([pScore, nScore])).argsort()
    rankvec = np.concatenate([np.ones(T), np.zeros(N)])
    rankvec = rankvec[IX]

    topK = np.minimum(atK, R)
    avp = (np.cumsum(rankvec) * rankvec) / np.arange(1, R + 1)
    avp = np.cumsum(avp)
    return avp[topK - 1] / np.minimum(topK, T)


def recallAtK(pScore, nScore, atK=None):
    """
    pScore: scores for positive instances
    nScore: scores for negative instances
    atK:    top K
    """
    pScore = np.asarray(pScore).flatten()
    nScore = np.asarray(nScore).flatten()
    T = len(pScore)
    N = len(nScore)
    R = T + N

    if atK is None: atK = T

    IX = (-np.concatenate([pScore, nScore])).argsort()
    rankvec = np.concatenate([np.ones(T), np.zeros(N)])
    rankvec = np.cumsum(rankvec[IX])

    topK = np.minimum(atK, R)
    hits = rankvec[topK - 1]
    return hits / T


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] for i, k in enumerate(d.keys())}


class NewsDataset(Dataset):

    def __init__(self, news_path):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(news_path,
                                         converters={
                                             'title': literal_eval,
                                             'abstract': literal_eval,
                                             'title_entities': literal_eval,
                                             'abstract_entities': literal_eval
                                         })

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        row = self.news_parsed.iloc[idx]
        item = {
            "id": row.id,
            "category": row.category,
            "subcategory": row.subcategory,
            "title": row.title,
            "abstract": row.abstract,
            "title_entities": row.title_entities,
            "abstract_entities": row.abstract_entities
        }
        return item


class UserDataset(Dataset):

    def __init__(self, behaviors_path, user2int_path):
        super(UserDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       # usecols=[0, 2],
                                       usecols=[1, 3],
                                       names=['user', 'clicked_news'])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0
        if model_name == 'LSTUR':
            print(f'User miss rate: {user_missed / user_total:.4f}')

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
                row.user,
            "clicked_news_string":
                row.clicked_news,
            "clicked_news":
                row.clicked_news.split()[:Config.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = Config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend(['PADDED_NEWS'] * repeated_times)

        return item


class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """

    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(
            behaviors_path,
            header=None,
            # usecols=range(4),
            # names=['user', 'time', 'clicked_news', 'impressions'])
            usecols=range(5),
            names=['ID', 'user', 'time', 'clicked_news', 'impressions'])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            'ID': row.ID,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item


@torch.no_grad()
def evaluate(model, directory, generate_json=False, json_path=None):

    news_dataset = NewsDataset(os.path.join(directory, 'news_parsed.tsv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=Config.batch_size,
                                 shuffle=False,
                                 num_workers=Config.num_workers,
                                 drop_last=False)

    news2vector = {}
    with tqdm(total=len(news_dataloader),
              desc="Calculating vectors for news") as pbar:
        for minibatch in news_dataloader:
            news_ids = minibatch["id"]
            if any(id not in news2vector for id in news_ids):
                news_vector = model.get_news_vector(minibatch)
                for id, vector in zip(news_ids, news_vector):
                    if id not in news2vector:
                        news2vector[id] = vector
            pbar.update(1)

    news2vector['PADDED_NEWS'] = torch.zeros(
        list(news2vector.values())[0].size())

    user_dataset = UserDataset(os.path.join(directory, 'behaviors.tsv'),
                               'data/train_l/user2int.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=Config.batch_size,
                                 shuffle=False,
                                 num_workers=Config.num_workers,
                                 drop_last=False)

    user2vector = {}
    with tqdm(total=len(user_dataloader),
              desc="Calculating vectors for users") as pbar:
        for minibatch in user_dataloader:
            user_strings = minibatch["clicked_news_string"]
            if any(user_string not in user2vector
                   for user_string in user_strings):
                clicked_news_vector = torch.stack([
                    torch.stack([news2vector[x].to(device) for x in news_list],
                                dim=0)
                    for news_list in minibatch["clicked_news"]
                ],
                    dim=0).transpose(0, 1)
                user_vector = model.get_user_vector(clicked_news_vector).squeeze()
                for user, vector in zip(user_strings, user_vector):
                    if user not in user2vector:
                        user2vector[user] = vector
            pbar.update(1)

    behaviors_dataset = BehaviorsDataset(
        os.path.join(directory, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=Config.num_workers)

    aucs = []
    mrrs = []
    ndcgs = []
    ndcg5s = []
    ndcg10s = []


    if generate_json:
        answer_file = open(json_path, 'w')
    with tqdm(total=len(behaviors_dataloader),
              desc="Calculating probabilities") as pbar:
        for minibatch in behaviors_dataloader:
            impression = {
                news[0].split('-')[0]: model.get_prediction(
                    news2vector[news[0].split('-')[0]],
                    user2vector[minibatch['clicked_news_string'][0]]).item()
                for news in minibatch['impressions']
                # val= torch.tensor([item.cpu().detach().numpy() for item in val]).cuda()

            }

            y_pred_list = list(impression.values())
            y_list = [
                int(news[0].split('-')[1]) for news in minibatch['impressions']
            ]

            auc = roc_auc_score(y_list, y_pred_list)
            mrr = mrr_score(y_list, y_pred_list)

            ndcg = ndcg_score(y_list, y_pred_list)
            ndcg5 = ndcg_score(y_list, y_pred_list, 5)
            ndcg10 = ndcg_score(y_list, y_pred_list, 10)

            aucs.append(auc)
            mrrs.append(mrr)
            ndcgs.append(ndcg)
            ndcg5s.append(ndcg5)
            ndcg10s.append(ndcg10)

            if generate_json:
                session = {
                    "uid": minibatch['user'][0],
                    "time": minibatch['time'][0],
                    "impression": value2rank(impression)
                }
                answer_file.write(json.dumps(session) + '\n')

            pbar.update(1)

    if generate_json:
        answer_file.close()

    return np.mean(aucs), np.mean(mrrs), \
           np.mean(ndcgs), np.mean(ndcg5s), np.mean(ndcg10s)


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Evaluating model {model_name}')
    model = Model(Config).to(device)
    from train1 import latest_checkpoint  # Avoid circular imports

    checkpoint_path = latest_checkpoint(
        os.path.join('./checkpoint', model_name))
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    auc, mrr, mrr5, mrr10, ndcg, ndcg5, ndcg10, \
    map, map5, map10, recall, recall5, recall10, \
    precision, precision5, precision10 = evaluate(model, 'data/test_l', True,
                                                  'data/test_l/prediction.json')
    print(
        f'AUC: {auc:.4f}\n'
        f'MRR: {mrr:.4f}\n'
        f'nDCG: {ndcg:.4f}\n'
        f'nDCG@5: {ndcg5:.4f}\n'
        f'nDCG@10: {ndcg10:.4f}\n'

    )
