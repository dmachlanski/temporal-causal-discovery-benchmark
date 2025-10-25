import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = '../workflow/results/synth_run1/summary/results.csv'
df = pd.read_csv(path)

metric = 'Recall'
lags = df[~df['lag'].isnull()]['lag'].unique()
n_lags = len(lags)
last_lag = lags[-1]
scenarios = ['indep', 'x_cause_y']
for s in scenarios:
    fig, axs = plt.subplots(nrows=1, ncols=n_lags, figsize=(24, 4), sharey=True)

    for l, ax in zip(lags, axs):
        df_local = df[(df['lag'] == l) & (df['seed'] == s)]
        plot_legend = last_lag == l
        sns.lineplot(data=df_local, x='subsample', y=metric, hue='id', ax=ax, legend=plot_legend)
        ax.set_xlabel('subsampling factor')
        ax.set_title(f"max lag = {int(l)}")
        if plot_legend:
            ax.get_legend().remove()

    fig.legend(ncol=3)
    fig.suptitle(f"scenario: {s}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{s}_recall.pdf", dpi=500)


metric = 'Precision'
lags = df[~df['lag'].isnull()]['lag'].unique()
n_lags = len(lags)
last_lag = lags[-1]
scenarios = ['indep', 'x_cause_y']
for s in scenarios:
    fig, axs = plt.subplots(nrows=1, ncols=n_lags, figsize=(24, 4), sharey=True)

    for l, ax in zip(lags, axs):
        df_local = df[(df['lag'] == l) & (df['seed'] == s)]
        plot_legend = last_lag == l
        sns.lineplot(data=df_local, x='subsample', y=metric, hue='id', ax=ax, legend=plot_legend)
        ax.set_xlabel('subsampling factor')
        ax.set_title(f"max lag = {int(l)}")
        if plot_legend:
            ax.get_legend().remove()

    fig.legend(ncol=3)
    fig.suptitle(f"scenario: {s}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{s}_precision.pdf", dpi=500)