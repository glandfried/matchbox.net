from tabnanny import verbose
#from surprise import SVDpp, Dataset, Reader
#from surprise.model_selection import cross_validate as surprise_cross_validate
#from surprise.model_selection import split
#from surprise.accuracy import rmse
from RatingPredictors import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
#import matchbox_refactor as mbox
from time import time
from hyperopt import hp, STATUS_OK

# https://surprise.readthedocs.io/en/stable/getting_started.html#use-a-custom-dataset
def infer_SVDpp(datasetName, ui, ts):
    reader = Reader(line_format="user item rating timestamp", sep=SEPARATOR, rating_scale=(0,5))
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

def infer_RandomForest(ttsi, n_estimators=100):
    X_train, X_test, y_train, y_test = ttsi.get()
    clf = RandomForestClassifier(random_state=1234, n_jobs=1, verbose=0, n_estimators=n_estimators, min_samples_split=2, min_samples_leaf=1).fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=clf.classes_)
    score_manual = evidence(y_test, y_pred_proba, labels=[0,1,2,3,4,5])
    return {"rmse":rmse, "cross-entropy":score}

def infer_NaiveBayes(ttsi):
    X_train, X_test, y_train, y_test = ttsi.get()
    clf = GaussianNB().fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=clf.classes_)
    return {"rmse":rmse, "cross-entropy":score}

def infer_matchbox_propio():
    MatchboxPropio.RecommenderSystem().Run()


class LGBM(Recommender):
    def name(self) -> str:
        return "LGBM"
    def defaultSpace(self) -> dict:
        return {
            #'boosting_type' : hp.choice('boosting_type', ["gbdt", "rf"]),
            'n_estimators': hp.quniform('n_estimators', 100, 500, 100),
            #"num_iterations": hp.choice("num_iterations", [100]),
            #"num_leaves": hp.quniform("num_leaves"),
            #'bagging_freq' : hp.choice('bagging_freq', range(10, 300, 10)),
            'subsample': hp.quniform('subsample', 0.7, 0.90, 0.02),
            #'objective': hp.choice('objective', ["regression", "regression_l1"]),
            'learning_rate': hp.qloguniform('learning_rate', np.log(0.04), np.log(0.17), 0.01),
            'reg_alpha': hp.choice('ra', [0, hp.quniform('reg_alpha', 0.01, 0.1, 0.01)]),
            #'reg_lambda': hp.choice('rl', [0, hp.quniform('reg_lambda', 0.01, 0.1, 0.01)]),
        }
    def objective(self, params: dict) -> dict:
        params['n_estimators'] = int(params['n_estimators'])
        #params['num_iterations'] = int(params['num_iterations'])
        X_train, X_test, y_train, y_test = self.ttsi.get()
        t0 = time()
        clf = LGBMClassifier(random_state = 1234, verbose=-1, **params).fit(X_train,y_train)
        train_time = time() - t0
        y_pred = clf.predict(X_test) #TODO: esta prediciendo siempre 3??? Revisar con un dataset mas grande que el de 50
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