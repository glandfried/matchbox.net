import InfernetWrapper
from InfernetWrapper import *
#from tabnanny import verbose
from surprise import SVDpp, Dataset, Reader
from surprise.model_selection import cross_validate as surprise_cross_validate
from surprise.model_selection import split
from surprise.accuracy import rmse
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as sklearn_cross_validate
import sklearn.metrics
from sklearn.metrics import make_scorer
from lightgbm import LGBMClassifier
#import matchbox_refactor as mbox
import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from time import time
import matplotlib.pyplot as plt
from datetime import datetime

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

class Res():
    def __init__(self, estimate):
        self.est = estimate

class TrainTestSplitInstance():
    def __init__(self, datasetName):
        df = pd.read_csv(datasetName, dtype="int", header=None, sep=SEPARATOR)
        X = df.iloc[:,[0,1,3]]
        y = df.iloc[:,2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def to_csv(self):
        dfTrain = pd.concat([self.X_train, self.y_train], axis=1, sort=False)
        dfTrain = dfTrain.reindex(columns=[0,1,2,3])
        dfTrain.to_csv('data/MovieLens/data_50_train.csv', header=False, index=False)
        dfTest = pd.concat([self.X_test, self.y_test], axis=1, sort=False)
        dfTest = dfTest.reindex(columns=[0,1,2,3])
        dfTest.to_csv('data/MovieLens/data_50_test.csv', header=False, index=False)

def infer_matchbox_propio():
    MatchboxPropio.RecommenderSystem().Run()

def infer_matchboxnet_single(datasetName, ui, ts, traitCount=5, iterationCount=20):
    # TODO: ver como hacer que tome timestamps, genere una historia que se itere en el tiempo

    # Parametros a optimizar: cantidad de features, prior, 
    # Ejemplo de https://dotnet.github.io/infer/userguide/Learners/Matchbox%20recommender/Learner%20API.html
    dataMapping = CsvMapping(SEPARATOR)
    recommender = MatchboxCsvWrapper.Create(dataMapping)
    # Settings: https://dotnet.github.io/infer/userguide/Learners/Matchbox/API/Setting%20up%20a%20recommender.html
    recommender.Settings.Training.TraitCount = traitCount;
    recommender.Settings.Training.IterationCount = iterationCount;
    recommender.Train(datasetName);

    # Modos prediccion: https://dotnet.github.io/infer/userguide/Learners/Matchbox/API/Prediction.html
    #print("Posterior rating para usuario 196 e item 302 con 5 estrellas");
    posterior = recommender.PredictDistribution();
    predRating = recommender.Predict(ui.user, ui.item);
    #print(",".join(map(str,posterior)));
    #recommendations = recommender.Recommend(rating.user, 10);
    #print(",".join(map(str,recommendations)));
    res = EstimatedRating(ui.user, ui.item, ts, predRating)
    res.posterior = {i.Key:i.Value for i in posterior} #Convert dotnet SortedDictionary to python dict
    return res
    '''
    print("Imple dotnet:")
    dotnet_posteriors = {0: 0.12841634145715716, 1: 0.14323634733764679, 2: 0.16840827972422545, 3: 0.2646673070766022, 4: 0.10161728482043446, 5: 0.19365443958393389}
    print(dotnet_posteriors);
    print("Diferencia con implementacion en dotnet:")
    dotnet_ranking= [242,327,234,603,1014,387,95,222,465,201]
    print(["{0:.20f}".format(abs(dotnet_posteriors[i]-posterior[i])) for i in range(len(posterior))])
    '''

def dotNetDistToArray(IDict):
    res =[]
    enum = IDict.Values.GetEnumerator()
    while (enum.MoveNext()):
        res.append(enum.Current)
    return res

def infer_matchboxnet(ttsi, traitCount=5, iterationCount=20):
    # TODO: refactor with batch operations
    dataMapping = CsvMapping(SEPARATOR)
    recommender = MatchboxCsvWrapper.Create(dataMapping)
    # Settings: https://dotnet.github.io/infer/userguide/Learners/Matchbox/API/Setting%20up%20a%20recommender.html
    recommender.Settings.Training.TraitCount = traitCount;
    recommender.Settings.Training.IterationCount = iterationCount;
    recommender.Train('data/MovieLens/data_50_train.csv');
    
    y_pred = []
    y_pred_proba = []
    _, X_test, _, y_test = ttsi.get()
    for _, x in tqdm.tqdm(X_test.iterrows(), total=len(X_test)):
        y_pred.append(recommender.Predict(str(x[0]), str(x[1])))
        pred_proba = dotNetDistToArray(recommender.PredictDistribution(str(x[0]), str(x[1])))
        y_pred_proba.append(pred_proba[1:])
    
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=[1,2,3,4,5])
    score_manual = evidence(y_test, y_pred_proba, labels=[1,2,3,4,5])
    #posterior = recommender.PredictDistribution(ui.user, ui.item);
    #predRating = recommender.Predict(ui.user, ui.item);
    return {"rmse":rmse, "cross-entropy":score}

# https://surprise.readthedocs.io/en/stable/getting_started.html#use-a-custom-dataset
def infer_SVDpp(datasetName, ui, ts):
    reader = Reader(line_format="user item rating timestamp", sep=SEPARATOR, rating_scale=(1,5))
    algo = SVDpp()
    data = Dataset.load_from_file(datasetName, reader=reader).build_full_trainset()

    algo.fit(data)
    #trainset, testset = split.train_test_split(data, test_size=0.25)
    #result = Res(algo.fit(trainset).test(testset))
    
    if isinstance(ui, ObservedRating):
        p = algo.predict(ui.user,ui.item, r_ui=ui.value)
    else:
        p = algo.predict(ui.user,ui.item)
    
    result = EstimatedRating(p.uid, p.iid, estimate=p.est)
    result.accuracy = rmse(p.est)
    result.evidence = None   # couldn't find a way to get evidence for surprise impl of SVDpp
    return result

def evidence(y_true, y_pred_proba, labels):
    y_true_idx = [labels.index(i) for i in y_true]
    #return np.prod([y_pred_proba[i][y_true_idx[i]] for i in range(len(y_pred_proba))])
    return -np.sum([np.log(y_pred_proba[i][y_true_idx[i]]) for i in range(len(y_pred_proba))])/len(y_pred_proba)

def infer_SVDpp(ttsi):
    X_train, X_test, y_train, y_test = ttsi.get() 
    reader = Reader(line_format="user item rating timestamp", sep=SEPARATOR, rating_scale=(1,5))
    algo = SVDpp()
    algo.fit()

def infer_RandomForest(ttsi, n_estimators=100):
    X_train, X_test, y_train, y_test = ttsi.get()
    clf = RandomForestClassifier(random_state=1234, n_jobs=1, verbose=0, n_estimators=n_estimators, min_samples_split=2, min_samples_leaf=1).fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=clf.classes_)
    score_manual = evidence(y_test, y_pred_proba, labels=[1,2,3,4,5])
    return {"rmse":rmse, "cross-entropy":score}

def infer_NaiveBayes(ttsi):
    X_train, X_test, y_train, y_test = ttsi.get()
    clf = GaussianNB().fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=clf.classes_)
    return {"rmse":rmse, "cross-entropy":score}

def infer_LGBM(ttsi, params={"n_estimators":100, "min_child_samples":1}):
    pass

class Recommender():
    def __init__(self, ttsi: TrainTestSplitInstance, space: dict=None):
        self.ttsi = ttsi
        self.space = space if space is not None else self.defaultSpace()
    def setSpace(self, space: dict) -> None:
        """Set parameter space for trials."""
        self.space = space
    def bestCandidates(self):
        """Perform trials over parameter space and return best candidates."""
        trials = Trials()
        best = fmin(lambda x: self.objective(x), self.space, algo=tpe.suggest, max_evals=16, trials=trials)
        return self._formatOutput(trials)
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
    def plotOverfitting(self, df, name=None):
        if name is None:
            name = self.name() + "_" + datetime.today().strftime('%Y%m%d_%H-%M-%S')
        plt.figure(figsize=(8,6))
        plt.scatter(df.tr_loss, df.loss, c=(df.loss-df.tr_loss)/df.loss*100)
        plt.title('Comparison of training and dev losses.\n Color corresponds to overfitting percentage')
        plt.colorbar()
        m = min(df.tr_loss.min(), df.loss.min())
        M = max(df.tr_loss.max(), df.loss.max())
        plt.plot([m, M], [m, M], 'k--')
        plt.xlabel('tr loss')
        plt.ylabel('dev loss')
        plt.grid()
        plt.savefig(f"./figs/{name}.png")
    def champion(self, df:pd.DataFrame) -> pd.DataFrame:
        """Candidate(s) that overfits the least out of best candidates."""
        return df[df.loss < df.loss.min() * 1.001].sort_values('tr_loss', ascending=False).head(30)
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

class LGBM(Recommender):
    def name(self) -> str:
        return "LGBM"
    def defaultSpace(self) -> dict:
        return {
            #'boosting_type' : hp.choice('boosting_type', ["gbdt", "rf"]),
            'n_estimators': hp.quniform('n_estimators', 100, 500, 100),
            "num_iterations": hp.choice("num_iterations", [100]),
            #'bagging_freq' : hp.choice('bagging_freq', range(10, 300, 10)),
            'subsample': hp.quniform('subsample', 0.7, 0.90, 0.02),
            #'objective': hp.choice('objective', ["regression", "regression_l1"]),
            'learning_rate': hp.qloguniform('learning_rate', np.log(0.04), np.log(0.17), 0.01),
            #'reg_alpha': hp.choice('ra', [0, hp.quniform('reg_alpha', 0.01, 0.1, 0.01)]),
            #'reg_lambda': hp.choice('rl', [0, hp.quniform('reg_lambda', 0.01, 0.1, 0.01)]),
        }
    def objective(self, params: dict) -> dict:
        params['n_estimators'] = int(params['n_estimators'])
        params['num_iterations'] = int(params['num_iterations'])
        X_train, X_test, y_train, y_test = self.ttsi.get()
        t0 = time()
        clf = LGBMClassifier(random_state = 1234, verbose=-1, **params).fit(X_train,y_train)
        train_time = time() - t0
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        y_train_pred_proba = clf.predict_proba(X_train)
        rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
        score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=clf.classes_)
        score_train = sklearn.metrics.log_loss(y_train, y_train_pred_proba, labels=clf.classes_)
        return dict(
            rmse=rmse,
            loss=score, 
            tr_loss=score_train,
            params=params,
            train_time=train_time,
            status=STATUS_OK
        )


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