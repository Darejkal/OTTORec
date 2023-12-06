import json
import copy
from torch.utils.data import IterableDataset
import numpy as np
import random
import sys
from multiprocessing import Queue, Process
def evaluate(model, dataset,num_sessions,item_num, args):

    NDCG = 0.0
    HT = 0.0
    valid_session = 0.0
    if num_sessions>1000:
        sessionIndexes = sorted(random.sample(range(0, num_sessions), 1000))
    else:
        sessionIndexes = range(0, num_sessions)
    for si in sessionIndexes:
        train,valid,test=dataset[si]
        if len(train) < 1 or len(test) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        
        seq[idx] = valid[0]
        idx -= 1
        for i in reversed(train):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train)
        rated.add(0)
        item_idx = [test[0]]
        for _ in range(100):
            t = np.random.randint(1, item_num + 1)
            while t in rated: t = np.random.randint(1, item_num + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[seq],item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_session += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_session % 100 == 0:
            print('.',end='')
            sys.stdout.flush()

    return NDCG / valid_session, HT / valid_session
def evaluate_valid(model, dataset:IterableDataset, num_sessions,item_num, args):

    NDCG = 0.0
    HT = 0.0
    valid_session = 0.0

    if num_sessions>1000:
        sessionIndexes = sorted(random.sample(range(0, num_sessions), 1000))
    else:
        sessionIndexes = range(0, num_sessions)
    for si in sessionIndexes:
        train,valid,test=dataset[si]
        if len(train) < 1 or len(valid) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train)
        rated.add(0)
        item_idx = [valid[0]]
        for _ in range(100):
            t = np.random.randint(1, item_num + 1)
            while t in rated: t = np.random.randint(1, item_num + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[seq], item_idx]])
        predictions = predictions[0] 

        rank = predictions.argsort().argsort()[0].item()

        valid_session += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_session % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_session, HT / valid_session


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(dataset:IterableDataset, sessionnum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        session = np.random.randint(1, sessionnum + 1)
        dataset_session=dataset[session][0]
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = dataset_session[-1]
        idx = maxlen - 1

        ts = set(dataset_session)
        for i in reversed(dataset_session[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (session, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))
class WarpSampler(object):
    def __init__(self, dataset, sessionnum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(dataset,
                                                      sessionnum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(int(2e9))
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()