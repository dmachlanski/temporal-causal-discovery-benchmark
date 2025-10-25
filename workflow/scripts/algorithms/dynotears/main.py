import sys
import time
import pandas as pd

from causalnex.structure.dynotears import from_pandas_dynamic

sys.path.append('../..')
from graph_utils import temporal_to_adjmat

def get_result(sm, data, tau_max):
    graph_dict = dict()
    for name in data.columns:
        graph_dict[name] = []
    
    tname_to_name_dict = dict()
    count_lag = 0
    idx_name = 0
    for tname in sm.nodes:
        tname_to_name_dict[tname] = data.columns[idx_name]
        if count_lag == tau_max:
            idx_name = idx_name +1
            count_lag = -1
        count_lag = count_lag +1

    for ce in sm.edges:
        c = ce[0]
        e = ce[1]
        tc = int(c.partition("lag")[2])
        te = int(e.partition("lag")[2])
        t = tc - te
        if (tname_to_name_dict[c], -t) not in graph_dict[tname_to_name_dict[e]]:
            graph_dict[tname_to_name_dict[e]].append((tname_to_name_dict[c], -t))

    return graph_dict

df = pd.read_csv(snakemake.input['data'])

t_start = time.time()

max_lag = snakemake.params['alg_opt']['lag']

model = from_pandas_dynamic(df, p=max_lag, 
                        w_threshold=snakemake.params['alg_opt']['w_thres'],
                        lambda_w=snakemake.params['alg_opt']['lambda_w'],
                        lambda_a=snakemake.params['alg_opt']['lambda_a'])

t_end = time.time()
t_delta = t_end - t_start

result = get_result(model, df, max_lag)

adjmat = temporal_to_adjmat(result, df.columns)
adjmat.to_csv(snakemake.output['pred'])

pd.DataFrame([t_delta], columns=['runtime']).to_csv(snakemake.output['info'], index=False)