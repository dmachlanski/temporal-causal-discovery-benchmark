import os
import pickle
import pandas as pd
from causalrivers.tools import graph_to_label_tensor, preprocess_data, remove_trailing_nans

def load_samples_graph(opt, data_opt):
    index_col = 'datetime'
    data = pickle.load(open(os.path.join(data_opt['path'], 'datasets', opt['scenario'], 'east.p'), "rb"))
    data = data[opt['seed']]

    Y = graph_to_label_tensor(data, human_readable=True)
    # above returns j->i; switch to i->j format
    Y = Y.T
    Y_names = list(Y.columns)

    # Get all required ts
    #unique_nodes = list(set([item for sublist in Y_names for item in sublist]))
    unique_nodes = list(set(Y_names))
    unique_nodes = (
        ([index_col] + [str(x) for x in unique_nodes])
        if index_col
        else [str(x) for x in unique_nodes]
    )
    
    # load required files
    data = pd.read_csv(
        os.path.join(data_opt['path'], data_opt['data_path']),
        index_col=index_col if index_col else None,
        usecols=unique_nodes,
    )

    # apply specify preprocessing to the data
    data = preprocess_data(data, resolution=opt['resolution'], subsample=opt['subsample'])
    
    single_sample = data[[str(m) for m in Y.columns]]
    #single_sample = data[Y_names]
    single_sample = remove_trailing_nans(single_sample)
        
    return single_sample, Y

data, graph = load_samples_graph(snakemake.params['opt'], snakemake.params['data_opt'])

# Save processed data
data.to_csv(snakemake.output['data'], index=False)
graph.to_csv(snakemake.output['graph'])