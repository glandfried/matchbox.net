from RatingPredictors import *
import InfernetWrapper
from InfernetWrapper import *
import pandasnet #For pandas DataFrames and Series to work as parameters for C# functions https://pypi.org/project/pandasnet/
from hyperopt import hp, STATUS_OK
from time import time
import sklearn.metrics
from InfernetWrapper import Gaussian

class Matchbox(Recommender):
    def __init__(self, ttsi: TrainTestSplitInstance, max_trials: int=100, space: dict=None, fromDataframe=False):
        self.fromDataframe = fromDataframe
        super().__init__(ttsi, max_trials, space)
    def name(self) -> str:
        return "Matchbox"
    def defaultSpace(self) -> dict:
        return {
            "traitCount" :  hp.quniform('traitCount', 4, 10, 1),
            "iterationCount" : hp.quniform('iterationCount', 5, 30, 5),
            "UserTraitFeatureWeightPriorVariance" : hp.choice('UserTraitFeatureWeightPriorVariance', [0.25,0.5,0.75,1]),
            "ItemTraitFeatureWeightPriorVariance" : hp.choice('ItemTraitFeatureWeightPriorVariance', [0.25,0.5,0.75,1]),
            "ItemTraitVariance" : hp.choice("ItemTraitVariance", [0.25,0.5,0.75,1]),
            "UserTraitVariance" : hp.choice("UserTraitVariance", [0.25,0.5,0.75,1]),
            "AffinityNoiseVariance" : hp.choice("AffinityNoiseVariance", [1.0])
        }
    
    def bestParams(self) -> dict:
        return {
            "ItemTraitFeatureWeightPriorVariance": 1,
            "ItemTraitVariance": 1,
            "UserTraitFeatureWeightPriorVariance": 1,
            "UserTraitVariance": 1,
            "AffinityNoiseVariance": 1,
            "iterationCount": 10,
            "traitCount": 5,
            "numLevels": 5
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
    
    def addEstimatedValues(self, df, user, thr=None, tra=None, bias=None):
        def formatNumToString(var):
            if np.isinf(var):
                var = "np.inf" if var > 0 else "-np.inf"
            else:
                var = f"{var:.5f}"
            return var

        def GaussianToString(g):
            if g.IsPointMass:
                return f"Gaussian.PointMass({formatNumToString(g.Point)})"
            else:
                return f"Gaussian({formatNumToString(g.GetMean())}, {formatNumToString(g.GetVariance())})"  
            
        if thr is not None:
            if type(thr[0]) == Gaussian:
                # If gaussian make column for mean and variance
                for i in range(len(thr)):
                    df.loc[user, f'Threshold_{i}'] = GaussianToString(thr[i])
            else:
                # Else put absolute values
                for i in range(len(thr)):
                    df.loc[user, f'Threshold_{i}'] = thr[i]

        if tra is not None:
            if type(tra[0]) == Gaussian:
                for i in range(len(tra)):
                    df.loc[user, f'Trait_{i}'] = GaussianToString(tra[i])
            else:
                for i in range(len(tra)):
                    df.loc[user, f'Trait_{i}'] = tra[i]

        if bias is not None:
            if type(bias) == Gaussian:
                df.loc[user, f'Bias'] = GaussianToString(bias)
            else:
                df.loc[user, f'Bias'] = bias
        return df

    def GetPosteriors(self, recommender, path):
        userPosteriors = recommender.GetPosteriorDistributions().Users
        itemPosteriors = recommender.GetPosteriorDistributions().Items

        estimated_users = pd.DataFrame.from_dict({"user": [int(x) for x in userPosteriors.Keys]}).sort_values("user").reset_index(drop=True)
        estimated_items = pd.DataFrame.from_dict({"item": [int(x) for x in itemPosteriors.Keys]}).sort_values("item").reset_index(drop=True)

        print("Generating user data...")
        for user in tqdm(userPosteriors.Keys, total=len(userPosteriors.Keys)):
            posteriors = userPosteriors.get_Item(user)
            #os.makedirs(path+"/users", exist_ok=True)
            #plotThresholds(posteriors.Thresholds, user, path=path+"/users", truth=generated_users.iloc[int(user)][userThresholdMask])
            #plotItemTraits(posteriors.Traits, user, isUser=True, path=path+"/users", truth=generated_users.iloc[int(user)][userTraitMask])
            estimated_users = self.addEstimatedValues(estimated_users, int(user), thr=posteriors.Thresholds, tra=posteriors.Traits, bias=posteriors.Bias)

        print("Generating item data...")
        for item in tqdm(itemPosteriors.Keys, total=len(itemPosteriors.Keys)):
            posteriors = itemPosteriors.get_Item(item)
            #os.makedirs(path+"/items", exist_ok=True)
            #plotItemTraits(posteriors.Traits, item, isUser=False, path=path+"/items", truth=generated_items.iloc[int(item)][itemTraitMask])
            estimated_items = self.addEstimatedValues(estimated_items, int(item), tra=posteriors.Traits, bias=posteriors.Bias)
        
        estimated_users.to_csv(f"{path}/user_estimated.csv", header=True, index=False)
        estimated_items.to_csv(f"{path}/item_estimated.csv", header=True, index=False)

        return estimated_users, estimated_items

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
        if "numLevels" in params and int(params["numLevels"])!=5: 
            dataMapping = DataframeMapping(1, int(params["numLevels"])) if self.fromDataframe is None else CsvMapping(1, int(params["numLevels"]), ",")
        else:
            # CSV mapping by default starts ratings at 0 so we need to set it to start at 1
            params["numLevels"] = 5
            dataMapping = DataframeMapping(1,5) if self.fromDataframe is None else CsvMapping(1,5,",")
        recommender = MatchboxCsvWrapper.Create(dataMapping)
        # Settings: https://dotnet.github.io/infer/userguide/Learners/Matchbox/API/Setting%20up%20a%20recommender.html
        recommender.Settings.Training.TraitCount = int(params["traitCount"])
        recommender.Settings.Training.IterationCount = int(params["iterationCount"])        
        recommender.Settings.Training.Advanced.UserTraitFeatureWeightPriorVariance = float(params["UserTraitFeatureWeightPriorVariance"])
        recommender.Settings.Training.Advanced.ItemTraitFeatureWeightPriorVariance = float(params["ItemTraitFeatureWeightPriorVariance"])
        recommender.Settings.Training.Advanced.ItemTraitVariance = float(params["ItemTraitVariance"])
        recommender.Settings.Training.Advanced.UserTraitVariance = float(params["UserTraitVariance"])
        recommender.Settings.Training.Advanced.AffinityNoiseVariance = float(params["AffinityNoiseVariance"])
        #recommender.Settings.Training.BatchCount = 2000
        if "lossFunction" in params:
            recommender.Settings.Prediction.SetPredictionLossFunction(params["lossFunction"])
        return recommender

    def predictionStats(self, params, train_time, y_pred, y_pred_proba, y_train_pred_proba, y_train, y_test):
        rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
        score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=self.labels(int(params["numLevels"])))
        score_train = sklearn.metrics.log_loss(y_train, y_train_pred_proba, labels=self.labels(int(params["numLevels"])))
        
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

    def predictionResults(self, params=None, pretrained_recommender=None):
        params = self.bestParams() if params is None else params
        if pretrained_recommender is None:
            recommender = self.createRecommender(params)
            train_time = self.train(recommender)
            print("Finished training")
        else:
            print("Using pretrained recommender")
            recommender = pretrained_recommender
            train_time = 0.0

        y_pred, y_pred_proba, y_train_pred_proba = self.predict(recommender)
        _, x_test, y_train, y_test = self.ttsi.get()

        dfTest = pd.concat([x_test, y_test], axis=1, sort=False)
        dfTest = dfTest.reindex(columns=[0,1,2,3])
        dfTest = dfTest.rename(columns={0:"userId",1:"movieId", 2:"y_test", 3:"timestamp"})
        dfTest["y_pred"] = y_pred
        dfTest["y_pred_proba"]=pd.Series(self.correct_probas(y_test, y_pred_proba, self.labels(int(params["numLevels"] if "numLevels" in params else 5))))
        for i in range(len(y_pred_proba[0])):
            dfTest[f"y_proba_{i+1}"] = [a[i] for a in y_pred_proba]

        dict = self.predictionStats(params, train_time, y_pred, y_pred_proba, y_train_pred_proba, y_train, y_test)
        dfStats = self._addOtherStats(pd.DataFrame.from_dict({k: [v] for k, v in dict.items()}))
        
        return dfTest, dfStats
    
    def labels(self, numLevels):
        return list(range(1,numLevels+1))
    
    def correct_probas(self, y_true, y_pred_proba, labels):
        y_true_idx = [labels.index(i) for i in y_true]
        #return np.prod([y_pred_proba[i][y_true_idx[i]] for i in range(len(y_pred_proba))])
        return [y_pred_proba[i][y_true_idx[i]] for i in range(len(y_pred_proba))]

    def objective(self, params: dict, return_pred: bool=False, recommender=None) -> dict:
        print(params)
        recommender = self.createRecommender(params) if recommender is None else recommender
        train_time = self.train(recommender)
        print("Finished training")

        y_pred, y_pred_proba, y_train_pred_proba = self.predict(recommender)
        _, _, y_train, y_test = self.ttsi.get()

        return self.predictionStats(params, train_time, y_pred, y_pred_proba, y_train_pred_proba, y_train, y_test)
        rmse = sklearn.metrics.mean_squared_error(y_test, y_pred)
        score = sklearn.metrics.log_loss(y_test, y_pred_proba, labels=self.labels(int(params["numLevels"])))
        score_train = sklearn.metrics.log_loss(y_train, y_train_pred_proba, labels=self.labels(int(params["numLevels"])))
        
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