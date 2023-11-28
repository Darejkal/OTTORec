from torch.utils.data import Dataset,IterableDataset
import os
import json
import math
import pandas as pd
from typing import List
class PagedData(IterableDataset):
    BATCHSIZE=20000
    def __init__(self, 
                 path:str,
                datarange:None|List=None,
                 ):
        self.path=path
        files=[os.path.splitext(f)[0] for r,_,files in os.walk(self.path) for f in files]
        last_file_index=int(sorted(files,key=lambda x: int(x))[-1])
        """
        Count number of file and check the last file index to ensure no gap (missing files).
        The number of file is then used to check the boundness of data_range.
        """
        count=0
        if last_file_index:
            count=last_file_index
            i=0
            with open(os.path.join(self.path,f"{count}.jsonl")) as lines:
                for _ in lines:
                    i+=1
            count=i+count*self.BATCHSIZE
        if len(files)!=math.ceil(count/self.BATCHSIZE):
            raise Exception(f"Number of session and number of file mismatched.\
                            \nSession count divided by BATCHSIZE (ceiling): {math.ceil(self._len/self.BATCHSIZE)}.\
                            \nLast File: {last_file_index}.\
                            \nNumber of file: {len(files)}.\
                            \nAre some files from the data directory missing?")
        if datarange==None:
            self.range=[0,count]
        else:
            assert datarange[0]>=0
            assert datarange[1]<count
            self.size=datarange


    def __read_session__(self,line:str):
        data_raw=json.loads(line)
        assert data_raw["events"]
        # data=data_raw["events"]
        data=[d["aid"] for d in data_raw["events"]]
        if len(data)<3:
            valid=[]
            test=[]
        else:
            valid=data[-2]
            test=data[-1]
            data=data[:-2]
        return data,valid,test
    
    def __getitem__(self, index):
        assert self.range[0]<=index<self.range[1]
        pageIndex=int(index/self.BATCHSIZE)
        lineIndex=index-pageIndex*self.BATCHSIZE
        with open(os.path.join(self.path,f"{pageIndex}.jsonl")) as dataFile:
            i=0
            while(i<lineIndex):
                next(dataFile,None)
                i+=1
            return self.__read_session__(next(dataFile))
    def __len__(self):
        return self.range[1]-self.range[0]
    def __iter__(self):
        """
        Using iterable to
        reducing I/O cost 
        and at the same time ensure closure of Context.
        Iterable begins from range[0] to range[1] specified by datarange.
        """
        first_page=math.ceil(self.range[0]/self.BATCHSIZE)
        second_page=math.ceil(self.range[1]/self.BATCHSIZE)
        number_of_skip_lines=self.range[0]-first_page*self.BATCHSIZE
        number_of_keep_lines=self.range[1]-second_page*self.BATCHSIZE

        with open(os.path.join(self.path,f"{first_page}.jsonl")) as dataFile:
            for _ in range(number_of_skip_lines):
                next(dataFile)
            for line in dataFile:
                yield self.__read_session__(line)
        
        for pageID in range(first_page+1,second_page-1):
            dataFile = open(os.path.join(self.path,f"{pageID}.jsonl"))
            for line in dataFile:
                yield self.__read_session__(line)
            dataFile.close()

        with open(os.path.join(self.path,f"{second_page}.jsonl")) as dataFile:
            for _ in range(number_of_keep_lines):
                yield self.__read_session__(next(dataFile))

    
class ClassicPagedData(Dataset):
    """
    Only used for data analyzing
    """
    BATCHSIZE=20000
    def __init__(self, path:str):
        self.path=path
        self.__read_data__()
    def __read_data__(self):
        files=[os.path.splitext(f)[0] for r,_,files in os.walk(self.path) for f in files]
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
    @classmethod
    def getPageIndex(cls,index:int):
        return int(index/cls.BATCHSIZE)
    @classmethod
    def getLineIndex(cls,index:int):
        return index-cls.getPageIndex(index)*cls.BATCHSIZE
    def __getitem__(self, index):
        pageIndex=self.getPageIndex(index)
        lineIndex=self.getLineIndex(index)
        with open(os.path.join(self.path,f"{pageIndex}.jsonl")) as dataFile:
            i=0
            while(i<lineIndex):
                next(dataFile,None)
                i+=1
            return json.loads(next(dataFile))
    def __len__(self):
        return self._len
    def __iter__(self):
        """
        Reducing I/O cost and ensure closure of Context
        """
        for pageID in range(math.ceil(self.__len__()/self.BATCHSIZE)):
            dataFile = open(os.path.join(self.path,f"{pageID}.jsonl"))
            for line in dataFile:
                yield json.loads(line)
            dataFile.close()

            
