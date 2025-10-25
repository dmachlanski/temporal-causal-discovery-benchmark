import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR


def make_human_readable(out, d):
    out = pd.DataFrame(out, columns=d.columns, index=d.columns)
    out = pd.concat([pd.concat([out], keys=["Cause"], axis=1)], keys=["Effect"])
    return out


def summary_transform(pred, opt):
    if opt == "max":
        prediction = pred.max(axis=2)
    elif opt == "mean":
        prediction = pred.mean(axis=2)
    return prediction


def var_baseline(d,cfg, human_readable=False):
    """
    Simple Granger based strategy that selects based on absolute parameter values.
    """
    n_vars = d.values.shape[-1]

    d.index = pd.DatetimeIndex(d.index.values,
                               freq=d.index.inferred_freq)
    
    # For sime constant ts this sometimes fails so we predict 0 if no model can be estimated.
    try:
        # fit var with appropriate max lags
        res = VAR(d).fit(cfg.max_lag)
        # convert to bool and throw away intersection
        # !In the context of rivers, negative correlation do not really make sense.
        # I guess trying both is fair
        pred = res.params[1:]
        
        if cfg.var_absolute_values:
            pred = np.abs(pred)

        # reformat to original caused causing lag:
        # :) einsum needed i guess
        pred = np.stack(
            [pred.values[:, x].reshape(cfg.max_lag, n_vars).T for x in range(pred.shape[1])]
        )
    except:
        pred = np.zeros((n_vars,n_vars,cfg.max_lag))
        print("Fitting failed")
    out = summary_transform(pred, cfg.map_to_summary_graph)
    

    if human_readable:
        out = make_human_readable(out, d)
    return out