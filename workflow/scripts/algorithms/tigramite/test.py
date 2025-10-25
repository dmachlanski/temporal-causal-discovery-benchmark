import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#import sklearn

#import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys

from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
#from tigramite.lpcmci import LPCMCI

from tigramite.independence_tests.parcorr import ParCorr
#from tigramite.independence_tests.robust_parcorr import RobustParCorr
#from tigramite.independence_tests.parcorr_wls import ParCorrWLS 
#from tigramite.independence_tests.gpdc import GPDC
#from tigramite.independence_tests.cmiknn import CMIknn
#from tigramite.independence_tests.cmisymb import CMIsymb
#from tigramite.independence_tests.gsquared import Gsquared
#from tigramite.independence_tests.regressionCI import RegressionCI

seed = 42
np.random.seed(seed)     # Fix random seed
def lin_f(x): return x
links_coeffs = {0: [((0, -1), 0.7, lin_f), ((1, -1), -0.8, lin_f)],
                1: [((1, -1), 0.8, lin_f), ((3, -1), 0.8, lin_f)],
                2: [((2, -1), 0.5, lin_f), ((1, -2), 0.5, lin_f), ((3, -3), 0.6, lin_f)],
                3: [((3, -1), 0.4, lin_f)],
                }


T = 1000     # time series length


data = pd.read_csv('../../../results/data/features~5/n~200/samples.csv').to_numpy()

dataframe = pp.DataFrame(data, datatime = {0:np.arange(len(data))})

tp.plot_timeseries(dataframe); plt.show()