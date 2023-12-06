from src.customlog import CustomLogger
from src.data import PagedData
from src.model import SASRec
from src.utils import evaluate,WarpSampler,evaluate_valid
import torch
import os
import time
import torch
import argparse
import numpy as np
import json
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--stats_file', required=True, type=str)
parser.add_argument('--max_iter', default=5000, type=int)

args = parser.parse_args()
if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)
with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    print(f"Writing args.txt to {args.train_dir}.")
f.close()
print("Args parsed sucessfully!")
def getNumBatch(dataset_len:int,batch_size:int,max_iter:int=-1):
    num_batch=int(len(dataset)/args.batch_size)
    if max_iter==-1:
        return num_batch
    if num_batch<max_iter:
        print("Num_batch is smaller than args.max_iter. Using num_batch in place of max_iter")
    else:
        num_batch=max_iter
    return num_batch
def loadStatsFile(stats_file:str):
    with open(stats_file) as f:
        stats=json.load(f)
    return stats["train"],stats["test"],stats["num_items"]
def createSampler(num_sessions:int,batch_size:int,max_len:int,n_workers=3):
    return WarpSampler(dataset, num_sessions, num_items, batch_size=batch_size, maxlen=max_len, n_workers=n_workers)
def createModel(num_items:int,args):
    model=SASRec(len(dataset),num_items,args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    return model
def saveModel(args):
    fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'\
        .format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
    torch.save(model.state_dict(), os.path.join(args.train_dir, fname))
    print(f"Epoch {epoch} saved----------------")
def loadPreviousModel(model:torch.nn.Module,state_dict_path:str):
    """
    Return epoch_start_idx (indexed from 1)
    """
    try:
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(args.device)))
        tail = state_dict_path[state_dict_path.find('epoch=') + 6:]
        return int(tail[:tail.find('.')]) + 1
    except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
        print('failed loading state_dicts, pls check file path: ', end="")
        print(state_dict_path)
        exit()
    finally:
        # in case of jupyter notebook => will train model as new.
        return 1
def printSequenceLen(dataset,logger:CustomLogger):
    if(len(dataset)>100000):
        logger.log("","Will not calculate average sequence length because dataset is too large!")
    else:
        cc = 0.0
        for session in dataset:
            cc += len(session.train.target)
        logger.log("",'Average sequence length: %.2f' % (cc / len(dataset)))
if __name__ == '__main__':
    logger=CustomLogger(log_file=os.path.join(args.train_dir, 'log.txt'))
    dataset=PagedData(args.dataset)
    num_batch=getNumBatch(len(dataset),args.batch_size,args.max_iter)
    printSequenceLen(dataset,logger)
    train_stats, test_stats, num_items=loadStatsFile(args.stats_file)
    sampler=createSampler(train_stats["num_sessions"],args.batch_size,args.maxlen)
    model=createModel(num_items,args)
    model.train()
    if args.state_dict_path is not None:
        epoch_start_idx=loadPreviousModel(model,args.state_dict_path)
    else:
        epoch_start_idx = 1
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, train_stats["num_sessions"], num_items, args)
        logger.log("INFERENCE",'test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]),True)
        exit()
    ce_criterion = torch.nn.CrossEntropyLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        print(f"Epoch {epoch}")
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model( seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = ce_criterion(pos_logits[indices], pos_labels[indices])
            loss += ce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            logger.log("LOSS","loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()),True) # expected 0.4~0.6 after init few epochs
    
        if epoch % 1 == 0:
            model.eval()
            print('Evaluating', end='')
            t_test = evaluate(model, dataset,train_stats["num_sessions"], num_items, args)
            t_valid = evaluate_valid(model, dataset,train_stats["num_sessions"], num_items, args)
            print()
            logger.log("EPOCH",'epoch:%d, valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            saveModel(args)
