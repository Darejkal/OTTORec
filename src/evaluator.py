import random
import sys

import numpy as np
import torch

from src.data import JSONLEventData
from typing import List,Tuple
def evaluate_recall(groundtruth:List[Tuple[int]], predicted:List[List[Tuple[int]]], k:int):
    recall=0
    length=len(groundtruth)
    assert length!=0
    assert length==len(predicted)
    for i in range(length):
        if groundtruth[i] in predicted[i][:k]:
            recall+=1
    return recall/length
def evaluate_MRR(groundtruth:List[Tuple[int]], predicted:List[List[Tuple[int]]], k:int):
    length=len(groundtruth)
    assert length!=0
    assert length==len(predicted)
    for i in range(length):
        if groundtruth[i] in predicted[i][:k]:
            return 1/i
    return 0
def calculate_ranks(logits, labels, cutoffs):
    num_logits = logits.shape[-1]
    k = min(num_logits, torch.max(cutoffs).item())
    _, indices = torch.topk(logits, k=int(k), dim=-1)
    indices = torch.flip(indices, dims=[-1])
    hits = indices == labels.unsqueeze(dim=-1)
    ranks = torch.sum(torch.cumsum(hits, -1), -1) - 1.
    ranks[ranks == -1] = float('inf')
    return ranks

def pointwise_recall(ranks, cutoffs, mask):
    res = ranks < cutoffs.unsqueeze(-1).unsqueeze(-1)
    return res.float() * mask
def pointwise_mrr(ranks, cutoffs, mask):
    res = torch.where(ranks < cutoffs.unsqueeze(-1).unsqueeze(-1), ranks, float('inf'))
    return (1 / (res + 1)) * mask
def mean_metric(pointwise_metric, mask):
    hits = torch.sum(pointwise_metric, dim=(2, 1))
    return hits / torch.sum(mask).clamp(0.0000005)