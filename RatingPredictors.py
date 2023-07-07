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
    return {"rmse":rmse, "cross-entropy":score, "manual cross-entropy":score_manual}

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
    clf = RandomForestClassifier(random_state=1234, n_jobs=1, n_estimators=n_estimators, min_samples_split=2, min_samples_leaf=1, verbose=0).fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=clf.classes_)
    score_manual = evidence(y_test, y_pred_proba, labels=[1,2,3,4,5])
    return {"rmse":rmse, "cross-entropy":score, "manual cross-entropy":score_manual}

def infer_NaiveBayes(ttsi):
    X_train, X_test, y_train, y_test = ttsi.get()
    clf = GaussianNB().fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=clf.classes_)
    return {"rmse":rmse, "cross-entropy":score}

def infer_LGBM(ttsi, n_estimators=100):
    X_train, X_test, y_train, y_test = ttsi.get()
    clf = LGBMClassifier(random_state = 1234, min_child_samples=1, n_estimators=n_estimators, verbose=-1).fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=clf.classes_)
    score_manual = evidence(y_test, y_pred_proba, labels=[1,2,3,4,5])
    return {"rmse":rmse, "cross-entropy":score, "manual cross-entropy":score_manual}