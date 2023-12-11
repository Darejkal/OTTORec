import json
import os
import numpy as np
import torch
from src.customlog import CustomLogger
from src.data import JSONLEventData
from src.model import ImprovisedSasrec
from src.utils import getNumBatch, saveModel
from typing import Type
from torch.utils.data import DataLoader,IterableDataset
config={
    "dataset":"data/otto/jsonl_processed",
    "train_dir":"output",
    "batch_size":128,
    "lr":0.001,
    "max_len":50,
    "hidden_size":50,
    "num_blocks":2,
    "num_epochs":201,
    "num_heads":1,
    "dropout_rate":0.5,
    "device":"cpu",
    "inference_only":False,
    "shuffle":True,
    "state_dict_path":None,
    "stats_file":"data/otto/jsonl_processed/stats.json",
    "num_batch":1,
    "num_batch_negatives": 127,
    "num_uniform_negatives": 16384,
    "reject_uniform_session_items": False,
    "reject_in_batch_items": True,
    "sampling_style": "batchwise",
}
assert 0 <= config["num_batch_negatives"] < config['batch_size']
def tryRestoreStateDict(model:torch.nn.Module,device:str,train_dir:str,state_dict_path:str):
    model.to(device)
    model.train()
    epoch_start_idx = 1
    if state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(device)))
            epoch=1
            with open(os.path.join(train_dir, 'save.json'),"r") as f:
                epoch=int(json.loads(next(f))["epoch"])
            epoch_start_idx= epoch
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(state_dict_path)
            exit()
        finally:
            # in case of jupyter notebook => will train model as new.
            return model,epoch_start_idx
    return model,epoch_start_idx
def jsonl_sample_func(result_queue,dataset:JSONLEventData,batch_size):
    def _sample():
        item=dataset[0]
        max_seqlen=dataset.max_seqlen
        pad=[0]*(max_seqlen-item["session_len"])
        e=pad+item["events"]
        l=pad+item["labels"]
        n=pad+[e[0] for e in item["negatives"]]
        return np.array(e),np.array(l),np.array(n)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(_sample())
        result_queue.put(zip(*one_batch))

def main():
    logger=CustomLogger(log_file=os.path.join(config["train_dir"], 'log.txt'))
    trainset=JSONLEventData(path=config["dataset"],
                            stats_file=config["stats_file"],
                            sub_category="train",
                            max_seqlen=config["max_len"],
                            num_in_batch_negatives=config["num_batch_negatives"],
                            num_uniform_negatives=config["num_uniform_negatives"],
                            reject_uniform_session_items=config["reject_uniform_session_items"],
                            reject_in_batch_items=config["reject_in_batch_items"],
                            sampling_style=config["sampling_style"])
    trainloader=DataLoader(trainset,
                              drop_last=True,
                              batch_size=config["batch_size"],
                              shuffle=config["shuffle"],
                              pin_memory=True,
                              persistent_workers=True,
                              num_workers=os.cpu_count() or 0,
                              collate_fn=trainset.dynamic_collate)
    testset=JSONLEventData(path=config["dataset"],
                           stats_file=config["stats_file"],
                           sub_category="test",
                           max_seqlen=config["max_len"],
                        num_in_batch_negatives=config["num_batch_negatives"],
                        num_uniform_negatives=config["num_uniform_negatives"],
                        reject_uniform_session_items=config["reject_uniform_session_items"],
                        reject_in_batch_items=config["reject_in_batch_items"],
                        sampling_style=config["sampling_style"])
    testloader=DataLoader(trainset,
                              drop_last=True,
                              batch_size=config["batch_size"],
                              shuffle=False,
                              pin_memory=True,
                              persistent_workers=True,
                              num_workers=os.cpu_count() or 0,
                              collate_fn=trainset.dynamic_collate)
    model=ImprovisedSasrec(trainset.num_items, config["max_len"],config["hidden_size"],config["dropout_rate"],config["num_heads"],config["sampling_style"],device=config["device"])
    model.to(model.device)
    _,epoch_start_idx=tryRestoreStateDict(model,config["device"],config["train_dir"],config["state_dict_path"])
    if config["inference_only"]:
        model.eval()
        score = model.evaluate()
        logger.log("INFERENCE",score,True)
        exit()
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98))
    logger=CustomLogger(os.path.join(config["train_dir"],"log.txt"))
    for epoch in range(epoch_start_idx, config["num_epochs"] + 1):
        logger.log("",f"Epoch {epoch}",True)
        for step in range(config["num_batch"]):
            batch=next(iter(trainloader))
            model.train_step(batch,step,logger,optimizer,bce_criterion)
        model.validate_step(next(iter(testloader)),epoch,logger,bce_criterion)


if __name__ == '__main__':
    main()