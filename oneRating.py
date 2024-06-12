from Matchbox import Matchbox
from RatingPredictors import TrainTestSplitInstance
import pandas as pd
from InfernetWrapper import Gaussian
import numpy as np

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

def addEstimatedValues(df, user, thr=None, tra=None, bias=None):
    if thr is not None:
        if type(thr[0]) == Gaussian:
            # If gaussian make column for mean and variance
            for i in range(len(thr)):
                df.loc[user, f'Threshold_{i}'] = GaussianToString(thr[i])
            #for i in range(len(thr)):
                #df.loc[user, f'Threshold_{i}_m'] = thr[i].GetMean()
                #df.loc[user, f'Threshold_{i}_v'] = thr[i].GetVariance()
        else:
            # Else put absolute values
            for i in range(len(thr)):
                df.loc[user, f'Threshold_{i}'] = thr[i]

    if tra is not None:
        if type(tra[0]) == Gaussian:
            # If gaussian make column for mean and variance
            for i in range(len(tra)):
                df.loc[user, f'Trait_{i}'] = GaussianToString(tra[i])
            #for i in range(len(tra)):
                #df.loc[user, f'Trait_{i}_m'] = tra[i].GetMean()
                #df.loc[user, f'Trait_{i}_v'] = tra[i].GetVariance()
        else:
            for i in range(len(tra)):
                df.loc[user, f'Trait_{i}'] = tra[i]

    if bias is not None:
        if type(bias) == Gaussian:
            # If gaussian make column for mean and variance
            df.loc[user, f'Bias'] = GaussianToString(bias)
            #df.loc[user, f'Bias_m'] = bias.GetMean()
            #df.loc[user, f'Bias_v'] = bias.GetVariance()
        else:
            df.loc[user, f'Bias'] = bias
    return df


dataset = f"./data/Tests/test4/ratings.csv"
ttsi = TrainTestSplitInstance(dataset) # No importa, le cargo cualquier cosa
ttsi.loadDatasets(preprocessed=True, NROWS=None, BATCH_SIZE=None)

mbox=Matchbox(ttsi, max_trials=1)
params = mbox.bestParams()
params["traitCount"] = 2
params["minRating"] = 0
params["maxRating"] = 1
recommender = mbox.createRecommender(params)
recommender.Settings.Training.Advanced.ItemBiasVariance = 1
recommender.Settings.Training.Advanced.UserBiasVariance = 1
recommender.Settings.Training.Advanced.AffinityNoiseVariance = 1
recommender.Settings.Training.Advanced.UserThresholdNoiseVariance = 0
recommender.Train(f"./data/Tests/oneRating/ratings_train.csv")
assert(recommender.Settings.Training.Advanced.UserThresholdNoiseVariance == 0), "UserThresholdNoiseVariance not set"
userPosteriors = recommender.GetPosteriorDistributions().Users
itemPosteriors = recommender.GetPosteriorDistributions().Items

estimated_users = pd.DataFrame.from_dict({"user": [int(x) for x in userPosteriors.Keys]}).sort_values("user").reset_index(drop=True)
estimated_items = pd.DataFrame.from_dict({"item": [int(x) for x in itemPosteriors.Keys]}).sort_values("item").reset_index(drop=True)

for user in userPosteriors.Keys:
    posteriors = userPosteriors.get_Item(user)
    estimated_users = addEstimatedValues(estimated_users, int(user), thr=posteriors.Thresholds, tra=posteriors.Traits, bias=posteriors.Bias)

for item in itemPosteriors.Keys:
    posteriors = itemPosteriors.get_Item(item)
    estimated_items = addEstimatedValues(estimated_items, int(item), tra=posteriors.Traits, bias=posteriors.Bias)
    
print(estimated_users.to_string())
estimated_users.to_csv(f"./data/Tests/oneRating/userEstimations.csv", header=True, index=True)
print(estimated_items.to_string())
estimated_items.to_csv(f"./data/Tests/oneRating/itemEstimations.csv", header=True, index=True)
#posterior = recommender.PredictDistribution(f"./data/Tests/test4/ratings_test.csv")#f"./data/Tests/oneRating/ratings.csv")
#print(posterior)

