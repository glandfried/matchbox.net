import InfernetWrapper
from InfernetWrapper import *
from System.Collections.Generic import Dictionary, SortedDictionary
import pandas as pd

def isDict(d):
    return hasattr(d, "GetType") and (d.GetType().GetGenericTypeDefinition() == clr.GetClrType(Dictionary) or d.GetType().GetGenericTypeDefinition() == clr.GetClrType(SortedDictionary))

def toDictRec(d):
    #res = {}
    #for v in d:
    #    res[v.Key] = toDict(v.Value) if isinstance(v.Value, IDictionary) else v.Value
    return {v.Key: toDictRec(v.Value) if isDict(v.Value) else v.Value for v in d}

def predDictToPandas2(d):
    res = []
    for userRatings in d:
        for movieRatings in userRatings.Value:
            res.append([v.Value for v in movieRatings.Value])
    return res

def predProbaDictToPandas(d):
    return [[r.Value for movieRatings in userRatings.Value for r in movieRatings.Value] for userRatings in d] 

def predDictToPandas(d):
    return [movieRating.Value for userRatings in d for movieRating in userRatings.Value] 

dataset = "./data/MovieLens/data_50_no_header.csv"
dataMapping = CsvMapping(",")
recommender = MatchboxCsvWrapper.Create(dataMapping)
# Settings: https://dotnet.github.io/infer/userguide/Learners/Matchbox/API/Setting%20up%20a%20recommender.html
recommender.Train(dataset);
pred = predDictToPandas(recommender.Predict(dataset))
pred_proba = predProbaDictToPandas(recommender.PredictDistribution(dataset))
print(pred)
print(pred_proba)