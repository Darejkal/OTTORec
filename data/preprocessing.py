import os
import json
from typing import Dict
from multiprocessing import Process,Queue
BATCHSIZE=20000
trainFile="train.jsonl"
trainOutFolder="train"
testFile="test.jsonl"
testOutFolder="test"

def filter_test_aids(train_sessions, test_sessions):
    train_aids = [event["aid"] for session in train_sessions for event in session["events"]]
    test_aids = [event["aid"] for session in test_sessions for event in session["events"]]
    aids_to_remove = set(test_aids).difference(set(train_aids))
    for session in test_sessions:
        session["events"] = [event for event in session["events"] if not event["aid"] in aids_to_remove]
    return (test_sessions, train_aids)

def write_stats(filename,*args):
    stats={}
    for field in args:
        assert type(field) is dict
        for k,v in field.items():
            stats[k]=v
    with open(filename, "w") as f:
        f.write(json.dumps(stats))

class DM_TypeMap():
    def __init__(self,type_map:Dict):
        self.type_map=type_map
        pass 
    def map(self,data)->bool:
        def _map(obj):
            obj["type"]=self.type_map[obj["type"]]
            return obj
        data["events"][:]=list(map(_map,data["events"]))
        return True
class StatedDM_AidMap():
    def __init__(self,target="aid",aid_map=dict(),filter=False):
        self.aid_nextID=0
        self.aid_map=aid_map
        self.target=target
        self.filter=filter
    def map(self,data)->bool:
        if self.filter:
            data["events"][:]=[e for e in data["events"] if e[self.target] in self.aid_map]
            for e in data["events"]:
                e[self.target]=self.aid_map[e[self.target]]
        else:
            for e in data["events"]:
                aid=e[self.target]
                if(aid not in self.aid_map):
                    self.aid_map[aid]=self.aid_nextID
                    self.aid_nextID+=1
                e[self.target]=self.aid_map[aid]
        return True
@DeprecationWarning
class DM_AidFilter():
    def __init__(self,target="aid",valid_values=[]):
        self.valid_values=set(valid_values)
        self.target=target
    def map(self,data)->bool:
        data["events"][:]=list(filter(lambda x: x[self.target] in self.valid_values,data["events"]))  # type: ignore
        return True
class DM_EventSizeFilter():
    def __init__(self, min_len:int):
        self.min_len=min_len
    def map(self,data)->bool:
        if len(data["events"])<self.min_len:
            return False
        return True
class StatedDM_SessionMapAndCount():
    def __init__(self):
        self.session_count=0
    def map(self,data)->bool:
        data["session"]=self.session_count
        self.session_count+=1
        return True
class StatedDM_EventCount():
    def __init__(self):
        self.event_count=0
    def map(self,data)->bool:
        self.event_count+=len(data["events"])
        return True
class DM_EventSort():
    def __init__(self,target="ts"):
        self.target=target
    def map(self,data)->bool:
        assert type(data["events"]) is list
        data["events"][:]=sorted(data["events"],key=lambda x: x[self.target])
        return True
def preprocess(line:str,data_maps=[])->str|None:
        data=json.loads(line)
        if type(data_maps) is list:
            for m in data_maps:
                # Modify inplace
                # Btw this decoupling causes significant slowdown in speed :( 
                flag=m.map(data)
                if not flag:
                    return None
        return json.dumps(data)
def handleData(outFolder:str,sourceFile:str,data_maps=[]):
    if not os.path.exists(f"{outFolder}"):
        os.mkdir(f"{outFolder}")
    print(f"Preprocessing")
    with open(sourceFile) as infile:
        flag=True
        pageID=0
        while flag:
            with open(f"{outFolder}/{pageID}.jsonl","+w") as outfile:
                print(pageID)
                for _ in range(BATCHSIZE):
                    line=next(infile,None)
                    if line:
                        nline=preprocess(line,data_maps)
                        if nline:
                            outfile.write(nline)
                            outfile.write("\n")
                    else:
                        flag=False
            pageID+=1
    print("Done preprocessing")

    
def main():
    cTypeMap,cAidMap,cEventSizeFilter,cEventSort=DM_TypeMap(type_map={"clicks":0,"carts":1,"orders":2}),StatedDM_AidMap("aid"),DM_EventSizeFilter(2),DM_EventSort(target="ts")
    tSessionMap,tEventCount= StatedDM_SessionMapAndCount(),StatedDM_EventCount()
    handleData(trainOutFolder,trainFile,[cEventSizeFilter,cEventSort,cTypeMap,cAidMap,tSessionMap,tEventCount])
    print(f"TRAIN => Sessions:{tSessionMap.session_count},Counts:{tEventCount.event_count}")

    num_items=len(cAidMap.aid_map)
    print(f"Num items: {num_items}")
    cAidMap.filter=True
    TSessionMap,TEventCount=StatedDM_SessionMapAndCount(),StatedDM_EventCount()
    handleData(testOutFolder,testFile,[
        cEventSizeFilter,
        cAidMap,
        cEventSizeFilter,
        cEventSort,
        cTypeMap,TSessionMap,TEventCount])
    print(f"TEST => Sessions:{TSessionMap.session_count},Counts:{TEventCount.event_count}")
    
    write_stats("stats.json", #filename
                {"num_items":num_items},
                {"train": {"num_sessions":tSessionMap.session_count,"num_events":tEventCount.event_count}},
                {"test": {"num_sessions":TSessionMap.session_count,"num_events":TEventCount.event_count}},
                {"aid_map":{"target":cAidMap.target,"map":cAidMap.aid_map}},
                )
    
if __name__=="__main__":
    main()