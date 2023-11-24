from torch.utils.data import Dataset
import os
import json
import math
class PagedData(Dataset):
    BATCHSIZE=20000
    def __init__(self, path:str):
        self.path=path
        files=[os.path.splitext(f)[0] for r,_,files in os.walk(path) for f in files]
        last_file_index=int(sorted(files,key=lambda x: int(x))[-1])

        count=0
        if last_file_index:
            count=last_file_index
            i=0
            with open(os.path.join(self.path,f"{count}.jsonl")) as lines:
                for _ in lines:
                    i+=1
            count=i+count*self.BATCHSIZE
        self._len=count
        if len(files)!=math.ceil(self._len/self.BATCHSIZE):
            raise Exception(f"Number of session and number of file mismatched.\
                            \nSession count divided by BATCHSIZE (ceiling): {math.ceil(self._len/self.BATCHSIZE)}.\
                            \nLast File: {last_file_index}.\
                            \nNumber of file: {len(files)}.\
                            \nAre some files from the data directory missing?")
    def __getitem__(self, index):
        pageIndex=int(index/self.BATCHSIZE)
        lineIndex=index-pageIndex*self.BATCHSIZE
        with open(os.path.join(self.path,f"{pageIndex}.jsonl")) as dataFile:
            i=0
            while(i<lineIndex):
                next(dataFile,None)
                i+=1
            return json.loads(next(dataFile))
    def __len__(self):
        return self._len