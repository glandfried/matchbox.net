from RatingPredictors import *
import InfernetWrapper
from InfernetWrapper import *
import pandasnet #For pandas DataFrames and Series to work as parameters for C# functions https://pypi.org/project/pandasnet/
from hyperopt import hp, STATUS_OK
from time import time
import sklearn.metrics

class Matchbox(Recommender):
    def __init__(self, ttsi: TrainTestSplitInstance, max_trials: int=100, space: dict=None, fromDataframe=False):
        self.fromDataframe = fromDataframe
        super().__init__(ttsi, max_trials, space)
    def name(self) -> str:
        return "Matchbox"
    def defaultSpace(self) -> dict:
        return {
            "traitCount" :  hp.quniform('traitCount', 10, 30, 1),
            "iterationCount" : hp.quniform('iterationCount', 5, 30, 5),
            "UserTraitFeatureWeightPriorVariance" : hp.choice('UserTraitFeatureWeightPriorVariance', [0.75,1,1.25,1.5,1.75,2]),
            "ItemTraitFeatureWeightPriorVariance" : hp.choice('ItemTraitFeatureWeightPriorVariance', [0,0.25,0.5,0.75,1]),
            "ItemTraitVariance" : hp.choice("ItemTraitVariance", [0,0.25,0.5,0.75,1]),
            "UserTraitVariance" : hp.choice("UserTraitVariance", [0,0.25,0.5,0.75,1])
        }
    
    def bestParams(self) -> dict:
        return {
            "ItemTraitFeatureWeightPriorVariance": 1,
            "ItemTraitVariance": 2,
            "UserTraitFeatureWeightPriorVariance": 1,
            "UserTraitVariance": 1,
            "iterationCount": 10,
            "traitCount": 20
            }
    def _formatPredDict(self, d):
        """
        Takes input of shape {usedId: {movieId: est_rating}} and converts it to array of estimated ratings.
            Example: {"12": {"1": 1, "2": 2}, "13": {"1": 3, "2": 4}} becomes [1,2,3,4]
            (where "12" and "13" are user ids, "1" and "2" are movie ids. Estimated ratings can be 1-5)
        """
        return [movieRating.Value for userRatings in d for movieRating in userRatings.Value] 
    def _formatPredProbaDict(self, d):
        """
        Takes input of shape {usedId: {movieId: {0: rating_proba, 1:rating_proba, ..., 5:rating_proba}}} and converts it to array of 5 rating arrays.
            Example: 
                {"12":{"1":{ 1: 0.3, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.1}, 
                    "2":{ 1: 0.2, 2: 0.3, 3: 0.2, 4: 0.2, 5: 0.1}},
                "13":{"1":{ 1: 0.3, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.1}, 
                    "2":{ 1: 0.2, 2: 0.3, 3: 0.2, 4: 0.2, 5: 0.1}}}
            (where "12" and "13" are user ids, "1" and "2" are movie ids)
            becomes:
            [[0.3, 0.2, 0.2, 0.2, 0.1],
            [0.2, 0.3, 0.2, 0.2, 0.1],
            [0.3, 0.2, 0.2, 0.2, 0.1],
            [0.2, 0.3, 0.2, 0.2, 0.1]]
        """
        return [[r.Value for r in movieRatings.Value] for userRatings in d for movieRatings in userRatings.Value] 

    def train(self, recommender):
        t0 = time()
        if self.ttsi.trainBatches is None:
            if self.fromDataframe:
                recommender.Train(MatchboxCsvWrapper.MakeTuple(self.ttsi.X_train, self.ttsi.y_train));
            else:
                recommender.Train(self.ttsi.trainCsvPath())
        else:
            if self.fromDataframe:
                raise NotImplementedError("Training in batches using dataframes not implemented.")
            else:
                for i in range(self.ttsi.trainBatches):
                    recommender.Train(self.ttsi.trainCsvPath(idx=i)); #TODO: this is not working?
        train_time = time() - t0
        return train_time
    
    def predict(self, recommender):
        y_pred = []
        y_pred_proba = []
        y_train_pred_proba = []
        if self.ttsi.testBatches is None:
            if self.fromDataframe:
                # Load without batching, from Dataframes
                y_pred += self._formatPredDict(recommender.Predict(MatchboxCsvWrapper.MakeTuple(self.ttsi.X_test, self.ttsi.y_test)))
                y_pred_proba += self._formatPredProbaDict(recommender.PredictDistribution(MatchboxCsvWrapper.MakeTuple(self.ttsi.X_test, self.ttsi.y_test)))
                y_train_pred_proba += self._formatPredProbaDict(recommender.PredictDistribution(MatchboxCsvWrapper.MakeTuple(self.ttsi.X_train, self.ttsi.y_train)))
            else:
                # Load without batching, from CSV
                y_pred += self._formatPredDict(recommender.Predict(self.ttsi.testCsvPath()))
                y_pred_proba += self._formatPredProbaDict(recommender.PredictDistribution(self.ttsi.testCsvPath()))
                y_train_pred_proba += self._formatPredProbaDict(recommender.PredictDistribution(self.ttsi.trainCsvPath()))
        else:
            if self.fromDataframe:    
                raise NotImplementedError("Training in batches using dataframes not implemented.")
            else:
                for i in range(self.ttsi.testBatches):
                    y_pred += self._formatPredDict(recommender.Predict(self.ttsi.testCsvPath(idx=i)))
                    y_pred_proba += self._formatPredProbaDict(recommender.PredictDistribution(self.ttsi.testCsvPath(idx=i)))
                    y_train_pred_proba += self._formatPredProbaDict(recommender.PredictDistribution(self.ttsi.trainCsvPath(idx=i)))
            
        return y_pred, y_pred_proba, y_train_pred_proba

    def createRecommender(self, params):
        dataMapping = DataframeMapping() if self.fromDataframe is None else CsvMapping()
        recommender = MatchboxCsvWrapper.Create(dataMapping)
        # Settings: https://dotnet.github.io/infer/userguide/Learners/Matchbox/API/Setting%20up%20a%20recommender.html
        recommender.Settings.Training.TraitCount = int(params["traitCount"])
        recommender.Settings.Training.IterationCount = int(params["iterationCount"])
        #recommender.Settings.Training.BatchCount = 2000
        return recommender

    def predictionResults(self):
        recommender = self.createRecommender(self.bestParams())
        train_time = self.train(recommender)
        print("Finished training")

        y_pred, y_pred_proba, y_train_pred_proba = self.predict(recommender)
        _, x_test, _, y_test = self.ttsi.get()
        dfTest = pd.concat([x_test, y_test], axis=1, sort=False)
        dfTest = dfTest.reindex(columns=[0,1,2,3])
        dfTest = dfTest.rename(columns={0:"userId",1:"movieId", 2:"y_test", 3:"timestamp"})
        dfTest["y_pred"] = y_pred
        for i in range(len(y_pred_proba[0])):
            dfTest[f"y_proba_{i}"] = [a[i] for a in y_pred_proba]

        return dfTest
    
    def labels(self):
        return [0,1,2,3,4,5]

    def objective(self, params: dict) -> dict:
        recommender = self.createRecommender(params)
        train_time = self.train(recommender)
        print("Finished training")

        y_pred, y_pred_proba, y_train_pred_proba = self.predict(recommender)
        _, _, y_train, y_test = self.ttsi.get()
        rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
        score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=self.labels()) #TODO: check we're getting 6 probas instead of 5. Other algos are using 1-5 labels i think
        score_train = sklearn.metrics.log_loss(y_train, y_train_pred_proba, labels=self.labels())
        
        #rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
        #score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=[1,2,3,4,5])
        #score_manual = evidence(y_test, y_pred_proba, labels=[1,2,3,4,5])
        #posterior = recommender.PredictDistribution(ui.user, ui.item);
        #predRating = recommender.Predict(ui.user, ui.item);
        return dict(
            rmse=rmse,
            loss=score, 
            tr_loss=score_train,
            params=params,
            train_time=train_time,
            status=STATUS_OK
        )