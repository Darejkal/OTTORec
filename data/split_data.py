import os
import json
count=0
batchsize=20000
sourceFile="train.jsonl"
outFolder="train"

if not os.path.exists(f"{outFolder}"):
    os.mkdir(f"{outFolder}")
_f = sorted(map(int,next(os.walk(f"{outFolder}"))[1]))
if (_f):
    _f=_f[-1]
    count=int(_f)-1
   # _f=sorted(next(os.walk(os.path.join("{outFolder}",_f)))[2])
   # if(_f):
   #     count=int(_f[-1])
print(f"Beginning from line: {count}")
def preprocess(line:str):
    data=json.loads(line)
    assert type(data["events"]) is list
    def event_map(obj):
        type_map={"clicks":0,"carts":1,"orders":2}
        obj["type"]=type_map[obj["type"]]
        obj.pop("ts",None)
        return obj
    data["events"]=sorted(data["events"],key=lambda x: x["ts"])
    data["events"]=list(map(event_map,data["events"]))
    return json.dumps(data)
with open(sourceFile) as infile:
    for _ in range(count*batchsize):
        next(infile)
    while True:
        with open(f"{outFolder}/{count}.jsonl","+w") as outfile:
            print(count)
            for i in range(batchsize):
                line=next(infile,None)
                if not line:
                    print("Done")
                    exit()
                line=preprocess(line)
                outfile.write(line)
                outfile.write("\n")
        count+=1

