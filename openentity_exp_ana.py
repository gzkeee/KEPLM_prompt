import copy
import random

import torch
from transformers import BertTokenizer, BertModel
from util import rel2id
from config import Config
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from util import load_file, load_json
import numpy as np

exp_num = 3
epoch_num = 5

ka = load_json('./data/OpenEntity/ana.json')
true_pred = load_json(f'./output_open_{exp_num}/true&pred_{epoch_num}.txt')
true, pred = true_pred['ture'], true_pred['pred']
label = load_file('./data/OpenEntity/labels.json')

print(true)
print(label.shape)

def accuracy(out, l):
    cnt = 0
    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        top = max(x1)
        for i in range(len(x1)):
            if x1[i] > 0:
                yy1.append(i)
            if x2[i] > 0:
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
        cnt += set(yy1) == set(yy2)
    return cnt, y1, y2


def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)

def loose_micro(true, pred):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in zip(true, pred):
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1(precision, recall)


def f():
    t_d1 = []
    p_d1 = []
    num = 0
    for i in range(len(true)):
        if ka['nk'][i] == 1:
            num += 1
            t_d1.append(true[i])
            p_d1.append(pred[i])

    print(loose_micro(t_d1, p_d1))
    print(num / len(true))


def cal_score(filter):
    print(filter)
    t_d1 = []
    p_d1 = []
    num = 0
    for i in range(len(filter)):
        if filter[i] == True:
            num += 1
            t_d1.append(true[i])
            p_d1.append(pred[i])

    print(loose_micro(t_d1, p_d1))
    print(num / len(true))


# print(label)
# cal_score(label[:, 0] == 1)

# print(true_pred)
# print(ka)
# print(loose_micro(true, pred))


