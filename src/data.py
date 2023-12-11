import numpy as np
import torch
from torch.utils.data import Dataset,IterableDataset
import os
import json
import math
import pandas as pd
from typing import List

from src.sampler import sample_in_batch_negatives, sample_uniform, sample_uniform_negatives_with_shape

class MultiFeaturedJSONLEventData(Dataset):
    BATCHSIZE=20000
    TYPE_MAP={"clicks":0,"carts":1,"orders":2}
    def __init__(self,path:str,stats_file:str,
                 max_seqlen=50,
                 sub_category="train",
                num_uniform_negatives=1,
                 num_in_batch_negatives=0,
                reject_uniform_session_items=False,
                 reject_in_batch_items=True,
                 sampling_style="eventwise",
                 datarange:None|List=None):
        self.path=os.path.join(path,sub_category+".jsonl")
        if(stats_file==None):
            raise NotImplemented
        else:
            self.stats=self.loadStatsFile(stats_file)
        self.num_sessions=self.stats[sub_category]["num_sessions"]
        self.num_items=self.stats["num_items"]
        self.num_events=self.stats[sub_category]["num_events"]
        if datarange==None:
            self.range=[0,self.num_sessions]
        else:
            assert datarange[0]>=0
            assert datarange[1]<=self.num_sessions
            self.range=datarange
        self.line_offsets = JSONLEventData.get_offsets(self.path)
        assert len(self.line_offsets) == self.num_sessions
        assert sampling_style in {"eventwise", "sessionwise", "batchwise"}
        self.max_seqlen=max_seqlen
        self.sampling_style=sampling_style
        self.num_uniform_negatives=num_uniform_negatives
        self.num_in_batch_negatives=num_in_batch_negatives
        self.reject_uniform_session_items=reject_uniform_session_items
        self.reject_in_batch_items=reject_in_batch_items
    def loadStatsFile(self,stats_file:str):
        with open(stats_file) as f:
            stats=json.load(f)
            # reduce ram cost
            del stats["aid_map"]
        return stats
    @staticmethod
    def get_offsets(jsonl_path:str):
            line_offsets = []
            with open(jsonl_path, "rt") as f:
                offset = 0
                for line_idx, line in enumerate(f):
                    line_len = len(line)
                    line_offsets.append((line_len, line_idx, offset))
                    offset += line_len
            line_offsets = [offset for _, _, offset in line_offsets]
            return line_offsets

    def __read_session__(self,line:str):
        data_raw=json.loads(line)
        assert data_raw["events"]
        events=data_raw["events"][-(self.max_seqlen+1):]

        etimes=[e["ts"] for e in events]
        ltimes=etimes[1:]
        etimes=etimes[:-1]
        etypes=[self.TYPE_MAP[e["type"]] for e in events]
        ltypes=etypes[1:]
        etypes=etypes[:-1]
        events=[e["aid"] for e in events]
        labels=events[1:]
        events=events[:-1]
        session_len=len(events)
        negatives=sample_uniform_negatives_with_shape(events, self.num_items, session_len, self.num_uniform_negatives, self.sampling_style, self.reject_uniform_session_items)
        return {
            "labels":labels,
            "positives":events,
            "negatives":negatives.tolist(),
            "features":{
                "ltimes":ltimes,
                "ltypes":ltypes,
                "etimes":etimes,
                "etypes":etypes,
            },
            "session_len":session_len
        }
    
    def __getitem__(self, index):
        assert self.range[0]<=index<self.range[1]
        with open(self.path) as dataFile:
            dataFile.seek(self.line_offsets[index])
            return self.__read_session__(dataFile.readline())
    def dynamic_collate(self, batch):
        batch_labels = list()
        batch_positives = list()
        batch_uniform_negatives = list()
        batch_mask = list()
        batch_session_len = list()
        batch_unpadded_positives = list()
        max_len = self.max_seqlen
        in_batch_negatives = list()
        features={}
        for k in batch["features"].keys():
            features[k]=list()
        for item in batch:
            session_len = item["session_len"]
            batch_positives.append((max_len - session_len) * [0] + item["positives"])
            for k in batch["features"].keys():
                features[k].append((max_len - session_len) * [0] + item["features"][k])
            batch_mask.append((max_len - session_len) * [0.] + session_len * [1.])
            batch_labels.append((max_len - session_len) * [0] + item["labels"])
            batch_session_len.append(session_len)
            batch_unpadded_positives.extend(item["positives"])

            if self.sampling_style=="eventwise":
                batch_uniform_negatives.append((max_len - session_len) * [[0]*self.num_uniform_negatives] + item["uniform_negatives"]) 
            elif self.sampling_style=="sessionwise":
                batch_uniform_negatives.append(item["uniform_negatives"]) 
            
        if self.sampling_style=="batchwise":
            batch_uniform_negatives = sample_uniform(self.num_items, [self.num_uniform_negatives], set(batch_unpadded_positives), self.reject_in_batch_items) 

        in_batch_negatives = sample_in_batch_negatives(batch_unpadded_positives, self.num_in_batch_negatives, batch_session_len, self.reject_in_batch_items) 
        return {
            'positives': torch.tensor(batch_positives, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long), 
            'mask': torch.tensor(batch_mask, dtype=torch.float),
            'session_len': torch.tensor(batch_session_len, dtype=torch.long),
            'in_batch_negatives': torch.tensor(in_batch_negatives, dtype=torch.long), 
            'uniform_negatives': torch.tensor(batch_uniform_negatives, dtype=torch.long),
            'features':{k: torch.tensor(v,dtype=torch.long) for k, v in features.items()}
        }   
    
    def __len__(self):
        return self.range[1]-self.range[0]
    # def __iter__(self):
    #     """
    #     Using iterable to reducing I/O cost while ensure closure of Context.
    #     Iterable begins from range[0] to range[1] specified by datarange.
    #     """
    #     with open(self.path) as dataFile:
    #         dataFile.seek(self.line_offsets[self.range[0]])
    #         for i in range(self.range[0],self.range[1]):
    #             yield self.__read_session__(next(dataFile))
class JSONLEventData(Dataset):
    BATCHSIZE=20000
    # TYPE_MAP={"clicks":0,"carts":1,"orders":2}
    def __init__(self,path:str,stats_file:str,
                 max_seqlen=50,
                 sub_category="train",
                num_uniform_negatives=1,
                 num_in_batch_negatives=0,
                reject_uniform_session_items=False,
                 reject_in_batch_items=True,
                 sampling_style="eventwise",
                 datarange:None|List=None):
        self.path=os.path.join(path,sub_category+".jsonl")
        if(stats_file==None):
            raise NotImplemented
        else:
            self.stats=self.loadStatsFile(stats_file)
        self.num_sessions=self.stats[sub_category]["num_sessions"]
        self.num_items=self.stats["num_items"]
        self.num_events=self.stats[sub_category]["num_events"]
        if datarange==None:
            self.range=[0,self.num_sessions]
        else:
            assert datarange[0]>=0
            assert datarange[1]<=self.num_sessions
            self.range=datarange
        self.line_offsets = JSONLEventData.get_offsets(self.path)
        assert len(self.line_offsets) == self.num_sessions
        assert sampling_style in {"eventwise", "sessionwise", "batchwise"}
        self.max_seqlen=max_seqlen
        self.sampling_style=sampling_style
        self.num_uniform_negatives=num_uniform_negatives
        self.num_in_batch_negatives=num_in_batch_negatives
        self.reject_uniform_session_items=reject_uniform_session_items
        self.reject_in_batch_items=reject_in_batch_items
    def loadStatsFile(self,stats_file:str):
        with open(stats_file) as f:
            stats=json.load(f)
            # reduce ram cost
            del stats["aid_map"]
        return stats
    @staticmethod
    def get_offsets(jsonl_path:str):
            line_offsets = []
            with open(jsonl_path, "rt") as f:
                offset = 0
                for line_idx, line in enumerate(f):
                    line_len = len(line)
                    line_offsets.append((line_len, line_idx, offset))
                    offset += line_len
            line_offsets = [offset for _, _, offset in line_offsets]
            return line_offsets

    def __read_session__(self,line:str):
        data_raw=json.loads(line)
        assert data_raw["events"]
        events=data_raw["events"][-(self.max_seqlen+1):]

        # etimes=[e["ts"] for e in events]
        # ltimes=etimes[1:]
        # etimes=etimes[:-1]
        # etypes=[self.TYPE_MAP[e["type"]] for e in events]
        # ltypes=etypes[1:]
        # etypes=etypes[:-1]
        events=[e["aid"] for e in events]
        labels=events[1:]
        events=events[:-1]
        session_len=len(events)
        negatives=sample_uniform_negatives_with_shape(events, self.num_items, session_len, self.num_uniform_negatives, self.sampling_style, self.reject_uniform_session_items)
        return {
            "labels":labels,
            "positives":events,
            "negatives":negatives.tolist(),
            # "features":{
            #     "ltimes":ltimes,
            #     "ltypes":ltypes,
            #     "etimes":etimes,
            #     "etypes":etypes,
            # },
            "session_len":session_len
        }
    
    def __getitem__(self, index):
        assert self.range[0]<=index<self.range[1]
        with open(self.path) as dataFile:
            dataFile.seek(self.line_offsets[index])
            return self.__read_session__(dataFile.readline())
    def dynamic_collate(self, batch):
        batch_labels = list()
        batch_positives = list()
        batch_uniform_negatives = list()
        batch_mask = list()
        batch_session_len = list()
        batch_unpadded_positives = list()
        max_len = self.max_seqlen
        in_batch_negatives = list()
        # features={}
        # for k in batch["features"].keys():
        #     features[k]=list()
        for item in batch:
            session_len = item["session_len"]
            batch_positives.append((max_len - session_len) * [0] + item["positives"])
            # for k in batch["features"].keys():
            #     features[k].append((max_len - session_len) * [0] + item["features"][k])
            batch_mask.append((max_len - session_len) * [0.] + session_len * [1.])
            batch_labels.append((max_len - session_len) * [0] + item["labels"])
            batch_session_len.append(session_len)
            batch_unpadded_positives.extend(item["positives"])

            if self.sampling_style=="eventwise":
                batch_uniform_negatives.append((max_len - session_len) * [[0]*self.num_uniform_negatives] + item["uniform_negatives"]) 
            elif self.sampling_style=="sessionwise":
                batch_uniform_negatives.append(item["uniform_negatives"]) 
            
        if self.sampling_style=="batchwise":
            batch_uniform_negatives = sample_uniform(self.num_items, [self.num_uniform_negatives], set(batch_unpadded_positives), self.reject_in_batch_items) 

        in_batch_negatives = sample_in_batch_negatives(batch_unpadded_positives, self.num_in_batch_negatives, batch_session_len, self.reject_in_batch_items) 
        return {
            'positives': torch.tensor(batch_positives, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long), 
            'mask': torch.tensor(batch_mask, dtype=torch.float),
            'session_len': torch.tensor(batch_session_len, dtype=torch.long),
            'in_batch_negatives': torch.tensor(in_batch_negatives, dtype=torch.long), 
            'uniform_negatives': torch.tensor(batch_uniform_negatives, dtype=torch.long),
            # 'features':{k: torch.tensor(v,dtype=torch.long) for k, v in features.items()}
        }   
    
    def __len__(self):
        return self.range[1]-self.range[0]
    # def __iter__(self):
    #     """
    #     Using iterable to reducing I/O cost while ensure closure of Context.
    #     Iterable begins from range[0] to range[1] specified by datarange.
    #     """
    #     with open(self.path) as dataFile:
    #         dataFile.seek(self.line_offsets[self.range[0]])
    #         for i in range(self.range[0],self.range[1]):
    #             yield self.__read_session__(next(dataFile))