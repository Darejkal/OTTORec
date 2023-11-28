import os
import json
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

def write_stats(num_items, num_train_sessions, num_train_events, num_test_sessions=None, num_test_events=None, filename=None):
    stats = {
        "train": {
            "num_sessions": num_train_sessions,
            "num_events": num_train_events
        },
        "num_items": num_items,
        "test": {
            "num_sessions": num_test_sessions,
            "num_events": num_test_events
        }
    }
    with open(filename, "w") as f:
        f.write(json.dumps(stats))

def preprocess(line:str,count:int,filter_lambda=None):
    data=json.loads(line)
    assert type(data["events"]) is list
    if len(data["events"])<2:
        return None,count,0
    def data_map(obj):
        type_map={"clicks":0,"carts":1,"orders":2}
        obj["type"]=type_map[obj["type"]]
        obj.pop("ts",None)
        return obj
    data["events"]=sorted(data["events"],key=lambda x: x["ts"])
    if filter_lambda:
        data["events"]=list(filter(filter_lambda,data["events"]))
    data["events"]=list(map(data_map,data["events"]))
    data["session"]=count
    return json.dumps(data),count+1,len(data["events"])
def handleData(outFolder:str,sourceFile:str,filter_lambda=None):
    if not os.path.exists(f"{outFolder}"):
        os.mkdir(f"{outFolder}")
    print(f"Preprocessing")
    with open(sourceFile) as infile:
        session_count=0
        event_count=0
        flag=True
        while flag:
            pageID=int(session_count/BATCHSIZE)
            with open(f"{outFolder}/{pageID}.jsonl","+w") as outfile:
                print(pageID)
                for _ in range(BATCHSIZE):
                    line=next(infile,None)
                    if line:
                        nline,session_count,num_event=preprocess(line,session_count,filter_lambda)
                        if nline:
                            event_count+=num_event
                            outfile.write(nline)
                            outfile.write("\n")
                    else:
                        flag=False
            
    print("Done preprocessing")
    return session_count,event_count

def main():
    # ts,te=handleData(trainOutFolder,trainFile)
    # print(f"Sessions:{ts},Counts:{te}")
    ts,te=12899779,216716096


    aid_set=set()
    page_size=int(ts/BATCHSIZE)
    for p in range(page_size):
        print(f"Counting unique aids: {p}/{page_size}")
        with open(f"{trainOutFolder}/{p}.jsonl","r") as f:
            for line in f:
                data=json.loads(line)
                for e in data["events"]:
                    aid_set.add(e["aid"])

    num_items=len(aid_set)
    print(f"Num items: {num_items}")
    Ts,Te=handleData(testOutFolder,testFile,lambda _x: _x["aid"] in aid_set)
    print(f"Sessions:{Ts},Counts:{Te}")
    write_stats(num_items,ts,te,Ts,Te,"stats.json")
if __name__=="__main__":
    main()