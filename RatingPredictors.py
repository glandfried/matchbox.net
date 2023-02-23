import InfernetWrapper
from InfernetWrapper import *
#from tabnanny import verbose
from surprise import SVDpp, Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import split
from surprise.accuracy import rmse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRanker
from lightgbm import LGBMClassifier
#import matchbox_refactor as mbox

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

def infer_matchbox_propio():
    MatchboxPropio.RecommenderSystem().Run()

def infer_matchboxnet(datasetName, ui, ts, traitCount=5, iterationCount=20):
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
    posterior = recommender.PredictDistribution(ui.user, ui.item);
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

def infer_RandomForest(datasetName, ui, ts, n_estimators=10):
    # TODO: agregar timestamp como feature, en los demas tb
    df = pd.read_csv(datasetName, dtype="int", header=None, sep=SEPARATOR)
    """
    pd.concat([df,pd.DataFrame([[rating.user, rating.item, rating.value, rating.timestamp]])])
    X = df.iloc[:,:2]
    y = df.iloc[:,2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf = RandomForestClassifier(random_state=1234, n_jobs=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1, verbose=10)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)
    """
    X = df.iloc[:,:2].to_numpy()
    y = df.iloc[:,2].to_numpy()
    clf = RandomForestClassifier(random_state=1234, n_jobs=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1, verbose=0)
    clf.fit(X, y)
    res = EstimatedRating(ui.user, ui.item, ts, clf.predict([[ui.user, ui.item]]))
    res.posterior = clf.predict_proba([[ui.user, ui.item]])
    return res

def infer_LightGBM(datasetName, ui, ts):
    df = pd.read_csv(datasetName, dtype="int", header=None, sep=SEPARATOR)
    X = df.iloc[:,:2].to_numpy()
    y = df.iloc[:,2].to_numpy()

    model = LGBMClassifier(min_child_samples=1, verbose=2)
    #https://tamaracucumides.medium.com/learning-to-rank-with-lightgbm-code-example-in-python-843bd7b44574
    query_train = [X.shape[0]]
    model.fit(X, y)
    res = EstimatedRating(ui.user, ui.item, ts, model.predict([[int(ui.user), int(ui.item)]]))
    res.posterior = model.predict_proba([[int(ui.user), int(ui.item)]])
    return res