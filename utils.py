# coding=utf-8
# @Time:2021/6/239:59
# @author: SinGaln
import scipy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from model import SimCSE
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer

model_class = {
    "bert":(BertConfig, SimCSE, BertTokenizer)
}

def load_tokenizer(args):
    return BertTokenizer.from_pretrained(args.bert_model_path)


def init_logger():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=logging.INFO)


def seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


def get_device(args):
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    return device

def compute(x,y):
    return scipy.stats.speramanr(x,y).correlation

def simcse_loss(y_pred, lamda=0.05):
    row = torch.arange(0, y_pred.shape[0], 3, device="cuda")
    col = torch.arange(y_pred.shape[0], device="cuda")
    col = torch.where(col % 3 != 0)[0].cuda()
    y_true = torch.arange(0, len(col), 2, device="cuda")
    similarity = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarity = torch.index_select(similarity, 0, row)
    similarity = torch.index_select(similarity, 1, col)
    similarity = similarity / lamda
    loss = torch.nn.functional.cross_entropy(similarity, y_true)
    return torch.mean(loss)

