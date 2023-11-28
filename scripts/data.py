from torch.utils.data import Dataset
import os
import json
import math
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import List
class PagedData(Dataset):
    BATCHSIZE=20000
    def __init__(self, 
                 path:str,
                 scale:List[str]=[],
                target_feature="aid",
                datarange:None|List=None,
                 ):
        """
        Scale is a list of cols to scale. For OTTO data, scaling is not preferable.
        Target_feature will be used to specify the feature that we will predict.
        """
        self.path=path

        """
        Whether or not to scale
        """
        self.scale=scale
        if scale:
            self.scaler=ColumnTransformer(
                [('scaler', StandardScaler(), scale)], 
                    remainder='passthrough')

        else:
            self.scaler=None
        self.__read_data__()
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

        """
        Feature that we want to predict.
        """
        self.target_feature=target_feature

    def __read_session__(self,line:str):
        df_raw=json.loads(line)
        assert df_raw["events"]
        df_raw=pd.DataFrame(df_raw["events"]).rename(columns={"ts":"date"})

        '''
        Reorder into
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        data=df_raw[1:]
        if self.scale:
            self.scaler.fit(data.values)
            data=self.scaler.transform(data.values)
        
        df_stamp=df_raw[["date"]]
        df_stamp["date"]=pd.to_datetime(df_stamp["date"])
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], 1).values
        return data,data_stamp
    
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
    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        else:
            return data
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
            for _ in number_of_skip_lines:
                next(dataFile)
            for line in dataFile:
                yield self.__read_session__(line)
        
        for pageID in range(first_page+1,second_page-1):
            dataFile = open(os.path.join(self.path,f"{pageID}.jsonl"))
            for line in dataFile:
                yield self.__read_session__(line)
            dataFile.close()

        with open(os.path.join(self.path,f"{second_page}.jsonl")) as dataFile:
            for _ in number_of_keep_lines:
                yield self.__read_session__(next(dataFile))

    
class ClassicPagedData(Dataset):
    """
    Only used for data analyzing
    """
    BATCHSIZE=20000
    def __init__(self, path:str):
        self.path=path
        self.scaler=StandardScaler()
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

            
