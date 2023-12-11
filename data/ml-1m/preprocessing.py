import pandas as pd
ratings=pd.read_csv("ratings.dat",header=None,sep="::",names=["UserID","MovieID","Rating","Timestamp"],engine="python")
ratings.sort_values("Timestamp")
for i in range(1,6041):
    ratings[ratings["UserID"]==i].to_csv(f"train/{i}.csv") 

import json
def write_stats(filename,*args):
    stats={}
    for field in args:
        assert type(field) is dict
        for k,v in field.items():
            stats[k]=v
    with open(filename, "w") as f:
        f.write(json.dumps(stats))
movies=pd.read_csv("movies.dat",header=None,sep="::",engine="python", encoding='latin-1')
write_stats("stats.json", #filename
            {"num_items":3952},
            {"train": {"num_sessions":6040,"num_events":ratings.shape[0]}},
            {"aid_map":{"target":"MovieID","map":dict(zip(movies.iloc[:,0].values.tolist(),movies.iloc[:,1].values.tolist()))}}
            )
