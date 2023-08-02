#import InfernetWrapper
#from InfernetWrapper import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as sklearn_cross_validate
import sklearn.metrics
#from sklearn.metrics import make_scorer
import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
#import os

SEPARATOR = ","

class UserItemPair():
    def __init__(self, user, item):
        self.user = user
        self.item = item

class ObservedRating(UserItemPair):
    def __init__(self, user, item, timestamp, value):
        self.value = value
        self.timestamp = timestamp
        super().__init__(user, item)

class EstimatedRating(UserItemPair):
    def __init__(self, user, item, timestamp, estimate):
        self.est = estimate
        self.timestamp = timestamp
        super().__init__(user, item)

class TrainTestSplitInstance():
    def __init__(self, datasetName):
        self.path = datasetName
        self.trainBatches = None
        self.testBatches = None

    def loadDatasets(self, preprocessed=True, NROWS=None, BATCH_SIZE=None):
        self.NROWS = NROWS
        self.BATCH_SIZE = BATCH_SIZE
        if preprocessed:
            self._loadDatasetsFromPreprocessedCsvs()
        else:
            self._loadDatasetsFromRawCsv()
            if self.BATCH_SIZE is not None:
                self.saveInBatches()   

    def saveInBatches(self):
        dfTrain = pd.concat([self.X_train, self.y_train], axis=1, sort=False)
        dfTrain = dfTrain.reindex(columns=[0,1,2,3])
        dfTest = pd.concat([self.X_test, self.y_test], axis=1, sort=False)
        dfTest = dfTest.reindex(columns=[0,1,2,3])
        i = 0
        print("=== Batching train set... ===")
        for offset in tqdm.trange(0, dfTrain.shape[0], self.BATCH_SIZE):
            dfTemp = dfTrain.loc[offset : offset+self.BATCH_SIZE-1]
            print(f"Saving batch {i} with offset {offset} and size {dfTemp.shape[0]} to {self.trainCsvPath(idx=i)}")
            dfTemp.to_csv(self.trainCsvPath(idx=i), header=False, index=False)
            i += 1
        self.trainBatches = i
        print(f"{i} train batches generated.")
        i = 0
        print("=== Batching test set... ===")
        for offset in tqdm.trange(0, dfTest.shape[0], self.BATCH_SIZE):
            dfTemp = dfTest.loc[offset : offset+self.BATCH_SIZE-1]
            print(f"Saving batch {i} with offset {offset} and size {dfTemp.shape[0]} to {self.testCsvPath(idx=i)}")
            dfTemp.to_csv(self.testCsvPath(idx=i), header=False, index=False)
            i += 1
        self.testBatches = i
        print(f"{i} test batches generated.")

    def _loadDatasetsFromRawCsv(self):
        df = pd.read_csv(self.path, sep=SEPARATOR, nrows=self.NROWS)
        self.size = df.shape[0]
        df["rating"] = df["rating"].round(0).astype(int)
        df = df.reindex(columns=["userId","movieId","rating","timestamp"])
        X = df.iloc[:,[0,1,3]]
        y = df.iloc[:,2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        self.X_train = X_train.rename(columns={"userId":0,"movieId":1,"timestamp":3}).reset_index(drop=True)
        self.X_test = X_test.rename(columns={"userId":0,"movieId":1,"timestamp":3}).reset_index(drop=True)
        self.y_train = y_train.rename(2).reset_index(drop=True)
        self.y_test = y_test.rename(2).reset_index(drop=True)
        self.to_csv()

    def _loadDatasetsFromPreprocessedCsvs(self):
        self.X_train, self.y_train, trainSize = self._generateDatasets(self.trainCsvPath())
        self.X_test, self.y_test, testSize = self._generateDatasets(self.testCsvPath())
        self.size = trainSize + testSize

    def _generateDatasets(self, path):
        df_train = pd.read_csv(path, sep=SEPARATOR, nrows=self.NROWS, header=None)
        return (df_train.iloc[:,[0,1,3]], df_train.iloc[:,2], df_train.shape[0])

    def get(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def trainCsvPath(self, idx=None):
        if idx is not None:
            return self.path.replace("ratings.csv",f"batches/ratings_train_{idx}.csv")
        return f"{self.path[:-4]}_train.csv"
    
    def testCsvPath(self, idx=None):
        if idx is not None:
            return self.path.replace("ratings.csv",f"batches/ratings_test_{idx}.csv")
        return f"{self.path[:-4]}_test.csv"

    def to_csv(self):
        dfTrain = pd.concat([self.X_train, self.y_train], axis=1, sort=False)
        dfTrain = dfTrain.reindex(columns=[0,1,2,3])
        dfTrain.to_csv(self.trainCsvPath(), header=False, index=False)
        dfTest = pd.concat([self.X_test, self.y_test], axis=1, sort=False)
        dfTest = dfTest.reindex(columns=[0,1,2,3])
        dfTest.to_csv(self.testCsvPath(), header=False, index=False)

class Recommender():
    def __init__(self, ttsi: TrainTestSplitInstance, space: dict=None):
        self.ttsi = ttsi
        self.space = space if space is not None else self.defaultSpace()

    def resultsName(self) -> str:
        """Root name used for results file of this recommender instance."""
        return self.name() + "_" + datetime.today().strftime('%Y%m%d_%H-%M-%S')

    def setSpace(self, space: dict) -> None:
        """Set parameter space for trials."""
        self.space = space

    def bestCandidates(self):
        """Perform trials over parameter space and return best candidates."""
        trials = Trials()
        best = fmin(lambda x: self.objective(x), self.space, algo=tpe.suggest, max_evals=16, trials=trials)
        best = self._formatOutput(trials)
        best["geo_mean"] = np.exp(-best["loss"]) #prediccion promedio, porque si en una productoria de las predicciones reemplazas todas las preds por el valor de la media geometrica, sale este
        if "tr_loss" in best.columns:
            best["geo_mean_tr"] = np.exp(-best["tr_loss"])
        best["dataset_size"] = self.ttsi.size
        best.to_csv(f"./trials/{self.resultsName()}.csv", header=True, index=False)
        return best
    
    def _formatOutput(self, trials):
        def flatten(doc, pref=''):
            res = {}
            for k, v in doc.items():
                k = f'{pref}.{k}' if pref else k
                if isinstance(v, dict):
                    res.update(flatten(v, k))
                else:
                    res[k] = v
            return res
        df = pd.DataFrame(list(map(flatten, [e['result'] for e in trials.trials])))
        return df.sort_values('loss')
    
    def champion(self, df:pd.DataFrame) -> pd.DataFrame:
        """Candidate(s) that overfits the least out of best candidates."""
        return df[df.loss < df.loss.min() * 1.001].sort_values('tr_loss', ascending=False).head(30)
    
    def plotOverfitting(self, best, name=None):
        name = name if name is not None else self.resultsName()
        plt.figure(figsize=(8,6))
        plt.scatter(best.tr_loss, best.loss, c=(best.loss-best.tr_loss)/best.loss*100)
        plt.title('Comparison of training and dev losses.\n Color corresponds to overfitting percentage')
        plt.colorbar()
        m = min(best.tr_loss.min(), best.loss.min())
        M = max(best.tr_loss.max(), best.loss.max())
        plt.plot([m, M], [m, M], 'k--')
        plt.xlabel('tr loss')
        plt.ylabel('dev loss')
        plt.grid()
        plt.savefig(f"./figs/{name}.png")

    def plotParamDistribution(self, best, name=None):
        name = name if name is not None else self.resultsName()
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
        
        param_names = list(self.space.keys())
        for param_name in param_names:
            plt.figure()
            visualize_param(param_name)
            plt.title(param_name)
            plt.tight_layout()
            plt.savefig(f"./figs/{name}_{param_name}.png")
    
    def name(self) -> str:
        """Readable name of recommender algorithm."""
        raise NotImplementedError("Subclasses should implement this")
    
    def defaultSpace(self) -> dict:
        """Default parameter space for trials."""
        raise NotImplementedError("Subclasses should implement this")
    
    def objective(self, params: dict) -> dict:
        """Function to call on every trial. 
           Return dictionary must include status and loss (metric to optimize)."""
        raise NotImplementedError("Subclasses should implement this")

"""
def objective(params):
    params['n_estimators'] = int(params['n_estimators'])
    print(params)
    cuts = [params['cut_publishers'], params['cut_developers'], params['cut_categories'], params['cut_tags']]
    pipe = make_pipeline(
        create_a_steam_pipe(cuts, params['n_components']),
        lgb.LGBMRegressor(random_state=42, **params)
    )
    t0 = time()
    pipe.fit(X_train, y_train)
    train_time = time() - t0
    loss=rmse(y_dev, pipe.predict(X_dev))
    print(f'loss {loss:.02f}')
    return dict(
        loss=loss,
        tr_loss=rmse(y_train, pipe.predict(X_train)), 
        params=params,
        train_time=train_time,
        status=STATUS_OK
    )

space = {
    'cut_publishers' : hp.choice('cut_publishers', [80,85,90,95,99]),
    'cut_developers' : hp.choice('cut_developers', [80,85,90,95,99]),
    'cut_categories' : hp.choice('cut_categories', [30,35,40,45,50,55,60,65]),
    'cut_tags' : hp.choice('cut_tags', [30,35,40,45,50,55,60,65]),
    'n_components' : hp.choice('n_components', [60,70,80,90,100,110,120,130]),
    #'boosting_type' : hp.choice('boosting_type', ["gbdt", "rf"]),
    'n_estimators': hp.quniform('n_estimators', 200, 500, 10),
    #'bagging_freq' : hp.choice('bagging_freq', range(10, 300, 10)),
    'subsample': hp.quniform('subsample', 0.7, 0.90, 0.02),
    #'objective': hp.choice('objective', ["regression", "regression_l1"]),
    'learning_rate': hp.qloguniform('learning_rate', np.log(0.04), np.log(0.17), 0.01),
    #'reg_alpha': hp.choice('ra', [0, hp.quniform('reg_alpha', 0.01, 0.1, 0.01)]),
    #'reg_lambda': hp.choice('rl', [0, hp.quniform('reg_lambda', 0.01, 0.1, 0.01)]),
}

trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, max_evals=16, trials=trials)
"""