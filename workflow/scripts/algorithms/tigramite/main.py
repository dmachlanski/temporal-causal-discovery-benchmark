import sys
import time
import pandas as pd

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI

from tigramite.independence_tests.parcorr import ParCorr

sys.path.append('../..')
from graph_utils import temporal_to_adjmat

def get_result(pcmci):
    res_dict = dict()
    for effect in pcmci.all_parents.keys():
        res_dict[pcmci.var_names[effect]] = []
        for cause, t in pcmci.all_parents[effect]:
            res_dict[pcmci.var_names[effect]].append((pcmci.var_names[cause], t))
    return res_dict

cond_test = ParCorr()

df = pd.read_csv(snakemake.input['data'])

lag_max = snakemake.params['alg_opt']['lag']

data_tig = pp.DataFrame(df.values, var_names=df.columns)

pcmci = PCMCI(dataframe=data_tig, cond_ind_test=cond_test, verbosity=0)

t_start = time.time()

if snakemake.params['param'] == 'pcmci':
    pcmci.run_pcmci(tau_min=1, tau_max=lag_max, pc_alpha=snakemake.params['alg_opt']['alpha'])
elif snakemake.params['param'] == 'pcmci_plus':
    pcmci.run_pcmciplus(tau_min=1, tau_max=lag_max, pc_alpha=snakemake.params['alg_opt']['alpha'])

t_end = time.time()
t_delta = t_end - t_start

result = get_result(pcmci)

adjmat = temporal_to_adjmat(result, df.columns)
adjmat.to_csv(snakemake.output['pred'])

pd.DataFrame([t_delta], columns=['runtime']).to_csv(snakemake.output['info'], index=False)