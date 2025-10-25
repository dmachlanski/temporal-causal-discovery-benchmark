import os
import sys

import numpy as np
import pandas as pd
import networkx as nx

sys.path.append('..')
from graph_utils import save_result_graph

def rename_lagged_cols(col):
    sp = col.split('_')
    return f"{sp[0]}_lag{sp[1]}"

def rename_underscored_cols(col):
    return col.replace('_', '-')

data_fname = os.path.join(snakemake.params['data_opt']['path'], f"{snakemake.params['opt']['seed']}.csv")
df = pd.read_csv(data_fname)

# One true graph for all data seeds
graph_fname = os.path.join(snakemake.params['data_opt']['path'], 'ground_truth.csv')
df_graph = pd.read_csv(graph_fname)

variables = ['hatch', 'load_in', 'load_out', 'pot_1', 'pot_2', 'current_in', 'current_out', 'pressure_downwind', 'pressure_upwind', 'rpm_in', 'rpm_out',
       'mic', 'pressure_intake', 'pressure_ambient', 'signal_1', 'signal_2']

# Select variables and normalize
df_data = df[variables]
for var in variables:
    b = df_data[var].max() - df_data[var].min()
    #b = df_data[var].std()
    if b != 0.0:
        a = df_data[var] - df_data[var].min()
        #a = df_data[var] - df_data[var].mean()
        df_data[var] = a / b

# Replace '_' with '-'
df_data.columns = df_data.columns.map(rename_underscored_cols)
df_graph.columns = df_graph.columns.map(rename_underscored_cols)

# Create lagged samples
n_lag = snakemake.params['opt']['lag']
lag_list = list(range(0, n_lag+1))
df_data_lagged = df_data.shift(periods=lag_list)[n_lag:]

df_data_lagged.columns = df_data_lagged.columns.map(rename_lagged_cols)

# Save processed data
df_data.to_csv(snakemake.output['data'], index=False)
df_data_lagged.to_csv(snakemake.output['data_lagged'], index=False)

# Save graph (df to DiGraph)
df_graph.index = df_graph.columns
gtrue = nx.from_pandas_adjacency(df_graph, create_using=nx.DiGraph)

save_result_graph(gtrue, snakemake.output['graphs'])