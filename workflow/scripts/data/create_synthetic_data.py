import sys
import networkx as nx

from dgp.src.data_generation_configs import (
    CausalGraphConfig, DataGenerationConfig, FunctionConfig, NoiseConfig, RuntimeConfig
)
from dgp.src.time_series_generator import TimeSeriesGenerator
from causalnex.structure.transformers import DynamicDataTransformer

sys.path.append('..')
from graph_utils import save_result_adjmat

def rename_remove_reorder_variables(adj_matrix, data):
    # Step 1: Remove 'Sx...' variables from the matrix
    adj_matrix = adj_matrix.loc[~adj_matrix.index.str.contains('Sx')]
    adj_matrix = adj_matrix.loc[:, ~adj_matrix.columns.str.contains('Sx')]

    # Step 2: Rename the variables so 't-1' becomes 'lag1', etc.
    def rename_variable(var_name):
        if var_name.endswith('_t'):
            return var_name.replace('_t', '_lag0')
        elif '_t-' in var_name:
            lag_num = var_name.split('_t-')[1]
            return var_name.split('_t-')[0] + '_lag' + lag_num
        else:
            return var_name

    adj_matrix.index = adj_matrix.index.map(rename_variable)
    adj_matrix.columns = adj_matrix.columns.map(rename_variable)

    # Step 3: Reorder the columns and rows to match the dataframe
    df_cols = data.columns.tolist()
    adj_matrix = adj_matrix.loc[df_cols, df_cols]
    return adj_matrix

def generate_data(complexity=30, min_lag=1, max_lag=3, num_targets=0, num_features=3, num_latent=0, num_samples=1000, seed=1, p_ar=0.3, p_edge=0.4):
    
    config = DataGenerationConfig(
        random_seed=seed,
        complexity=complexity,
        percent_missing=0.0,
        causal_graph_config=CausalGraphConfig(
            graph_complexity=complexity,
            include_noise=True,
            max_lag=max_lag,
            min_lag=min_lag,
            num_targets=num_targets,
            num_features=num_features,
            num_latent=num_latent,
            prob_edge=p_edge,
            max_parents_per_variable=3,
            max_target_parents=2,
            max_target_children=0,
            max_feature_parents=3,
            max_feature_children=2,
            max_latent_parents=2,
            max_latent_children=2,
            allow_latent_direct_target_cause=False,
            allow_target_direct_target_cause=False,
            prob_target_autoregressive=0.1,
            prob_feature_autoregressive=p_ar,
            prob_latent_autoregressive=0.2,
            prob_noise_autoregressive=0.0,
        ),
        function_config=FunctionConfig(
            functions=['piecewise_linear', 'trigonometric'], #['linear', 'piecewise_linear', 'monotonic', 'trigonometric']
            prob_functions=[0.5, 0.5],
        ),
        noise_config=NoiseConfig(
            distributions=['gaussian'],
            prob_distributions=[1.0],
            noise_variance=[0.01, 0.05]
        ),
        runtime_config=RuntimeConfig(
            num_samples=num_samples,
            data_generating_seed=seed
        )
    )

    ts_generator = TimeSeriesGenerator(config=config)
    datasets, causal_graph = ts_generator.generate_datasets()
    df = datasets[0]
    df_lags = DynamicDataTransformer(p=max_lag).fit_transform(df, return_df=True)
    
    # save data
    true_causal_graph_df = nx.to_pandas_adjacency(causal_graph.causal_graph)
    true_causal_graph_df = rename_remove_reorder_variables(true_causal_graph_df, df_lags)

    return true_causal_graph_df, df, df_lags


adjmat, data, data_lagged = generate_data(num_features=int(snakemake.params['opt']['features']), 
                                        num_samples=int(snakemake.params['opt']['n']),
                                        min_lag=int(snakemake.params['opt']['n_lag']),
                                        max_lag=int(snakemake.params['opt']['n_lag']),
                                        seed=int(snakemake.params['opt']['seed']),
                                        p_ar=float(snakemake.params['opt']['prob_ar']),
                                        p_edge=float(snakemake.params['opt']['prob_edge']))

adjmat.to_csv(snakemake.output['adjmat'], index=False)
data.to_csv(snakemake.output['data'], index=False)
data_lagged.to_csv(snakemake.output['data_lagged'], index=False)

save_result_adjmat(adjmat, data.columns, snakemake.output['graphs'])