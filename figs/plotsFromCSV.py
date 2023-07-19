import pandas as pd
import matplotlib.pyplot as plt

def plotOverfitting(best, name):
        best["loss"] = best["loss"].astype(float)
        best["tr_loss"] = best["tr_loss"].astype(float)
        best["c"] = (best.loss-best.tr_loss)/best.loss*100
        #best = best.quantile(0.95, numeric_only=True)
        plt.figure(figsize=(8,6))
        plt.scatter(best.tr_loss, best.loss, c=best.c)
        plt.title('Comparison of training and dev losses.\n Color corresponds to overfitting percentage')
        plt.colorbar()
        m = min(best.tr_loss.min(), best.loss.min())
        M = max(best.tr_loss.max(), best.loss.max())
        plt.plot([m, M], [m, M], 'k--')
        plt.xlabel('tr loss')
        plt.ylabel('dev loss')
        plt.grid()
        plt.savefig(f"./figs/{name}.png")

def plotParamDistribution(best, name):
    cut_point = best.loss.median()
    best_models_df = best[best.loss <= cut_point]
    worst_models_df = best[best.loss > cut_point]
    def visualize_param(param_name):
        s = best[f'params.{param_name}']
        if s.dtype.name == 'object':
            visualize_categorical_param(param_name)
        else: # assume numerical
            visualize_numerical_param(param_name)

    def visualize_categorical_param(param_name):
        pd.concat([
            best_models_df[f'params.{param_name}'].value_counts().rename('best'),
            worst_models_df[f'params.{param_name}'].value_counts().rename('worst')
        ], axis=1).plot.bar()

    def visualize_numerical_param(param_name):
        plt.violinplot([
            best_models_df[f'params.{param_name}'],
            worst_models_df[f'params.{param_name}']
        ])
        plt.xticks([1, 2], ['best', 'worst'])
    
    param_names = list(map(lambda y: y.replace("params.",""), filter(lambda x: "params." in x, best.columns)))
    for param_name in param_names:
        plt.figure()
        visualize_param(param_name)
        plt.title(param_name)
        plt.tight_layout()
        plt.savefig(f"./figs/{name}_{param_name}.png")

name="LGBM_20230718_17-28-55"
df = pd.read_csv(f"./trials/{name}.csv")
plotOverfitting(df, name)
#plotParamDistribution(df, name)