import argparse
import json
import copy
import os
import torch
from torch.utils.data import IterableDataset
import numpy as np
import random
import sys
from multiprocessing import Queue, Process


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
# def parseArgs():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', required=True)
#     parser.add_argument('--train_dir', required=True)
#     parser.add_argument('--batch_size', default=128, type=int)
#     parser.add_argument('--lr', default=0.001, type=float)
#     parser.add_argument('--max_len', default=50, type=int)
#     parser.add_argument('--hidden_size', default=50, type=int)
#     parser.add_argument('--num_blocks', default=2, type=int)
#     parser.add_argument('--num_epochs', default=201, type=int)
#     parser.add_argument('--num_heads', default=1, type=int)
#     parser.add_argument('--dropout_rate', default=0.5, type=float)
#     parser.add_argument('--l2_emb', default=0.0, type=float)
#     parser.add_argument('--device', default='cpu', type=str)
#     parser.add_argument('--inference_only', default=False, type=str2bool)
#     parser.add_argument('--shuffle', default=True, type=str2bool)
#     parser.add_argument('--state_dict_path', default=None, type=str)
#     parser.add_argument('--stats_file', required=True, type=str)
#     parser.add_argument('--num_batch', default=5000, type=int)
#     args = parser.parse_args()
#     if not os.path.isdir(args.train_dir):
#         os.makedirs(args.train_dir)
#     with open(os.path.join(args.train_dir, 'args.json'), 'w') as f:
#         f.write(json.dumps(vars(args)))
#         print(f"Writing args.json to {args.train_dir}.")
#     print("Args parsed sucessfully!")
#     return args
def getNumBatch(dataset_len:int,batch_size:int,max_iter:int=-1):
    num_batch=int(dataset_len/batch_size)
    if max_iter==-1:
        return num_batch
    if num_batch<max_iter:
        print("Num_batch is smaller than args.max_iter. Using num_batch in place of max_iter")
    else:
        num_batch=max_iter
    return num_batch
def saveModel(model:torch.nn.Module,epoch,args):
    with open(os.path.join(args.train_dir, 'save.json'), 'w') as f:
        f.write(json.dumps({"epoch":epoch}))
    torch.save(model.state_dict(), os.path.join(args.train_dir, "latest.pth"))
    print(f"Epoch {epoch} saved----------------")