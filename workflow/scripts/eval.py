import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

graph_true = pd.read_csv(snakemake.input['g_true'])
graph_hat = pd.read_csv(snakemake.input['g_hat'])
df_info = pd.read_csv(snakemake.input['info'])

cols = ['id']
vals = [snakemake.params['id']]

for key, value in snakemake.params['data_opt'].items():
    cols.append(key)
    vals.append(value)

for key, value in snakemake.params['alg_opt'].items():
    cols.append(key)
    vals.append(value)

cols.append('repeat')
vals.append(snakemake.params['repeats']['repeats'])

arr_true = graph_true.values.astype(bool)
arr_hat = graph_hat.values.astype(bool)

# we don't care about autoregressive links (inplace)
np.fill_diagonal(arr_true, 0)
np.fill_diagonal(arr_hat, 0)

arr_true = arr_true.flatten()
arr_hat = arr_hat.flatten()

metric_names = ['AUROC', 'F1', 'Recall', 'Precision']
metric_funcs = [roc_auc_score, f1_score, recall_score, precision_score]

for name, func in zip(metric_names, metric_funcs):
    score = func(arr_true, arr_hat)
    cols.append(name)
    vals.append(score)

df = pd.DataFrame([vals], columns=cols)
df = pd.concat([df, df_info], axis=1)

df.to_csv(snakemake.output[0], index=False)