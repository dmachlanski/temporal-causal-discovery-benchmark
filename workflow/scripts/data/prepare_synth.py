import os
import pandas as pd

data = pd.read_csv(os.path.join(snakemake.params['data_opt']['path'], snakemake.params['opt']['seed'], 'data.csv'), header=None)
graph = pd.read_csv(os.path.join(snakemake.params['data_opt']['path'], snakemake.params['opt']['seed'], 'graph.csv'), header=None)

cols = ['x', 'y']

# drop the index, don't need it
data = data.drop(columns=0)

# name columns
data.columns = cols
graph.columns = cols
graph.index = cols

# subsampling factor k
k = int(snakemake.params['opt']['subsample'])
if k > 1:
    data = data[::k] # take every k-th element

# Save processed data
data.to_csv(snakemake.output['data'], index=False)
graph.to_csv(snakemake.output['graph'])