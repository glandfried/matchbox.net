from Matchbox import Matchbox
from RatingPredictors import TrainTestSplitInstance
from Others import LGBM
from Others import RandomForest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = "./data/MovieLens/ml-100k/ratings.csv"
#dataset = "./data/Simulation/20240114_02-11-20_inferUpdate/ratings.csv"
ttsi = TrainTestSplitInstance(dataset)
ttsi.loadDatasets(preprocessed=True, NROWS=None, BATCH_SIZE=None)

print("Run experiments for 1 to 5 star 100k movielens....")
mbox=Matchbox(ttsi, max_trials=10)
mbox.minRating = 1
mbox.maxRating = 5
df_mbox=mbox.bestCandidates()
print("Matchbox", df_mbox)

rf=RandomForest(ttsi, max_trials=10)
df_rf=rf.bestCandidates()
print("RandomForest", df_rf)

lgbm=LGBM(ttsi, max_trials=10)
df_lgbm=lgbm.bestCandidates()
print("LGBM", df_lgbm)

print("Run experiments for binary 100k movielens....")
dataset = "./data/MovieLens/ml-100k-binary/ratings.csv"
#dataset = "./data/Simulation/20240114_02-11-20_inferUpdate/ratings.csv"
ttsi = TrainTestSplitInstance(dataset)
ttsi.loadDatasets(preprocessed=True, NROWS=None, BATCH_SIZE=None)
mbox=Matchbox(ttsi, max_trials=10)
mbox.minRating = 0
mbox.maxRating = 1
df_mbox=mbox.bestCandidates()
print("Matchbox", df_mbox)

rf=RandomForest(ttsi, max_trials=10)
df_rf=rf.bestCandidates()
print("RandomForest", df_rf)

lgbm=LGBM(ttsi, max_trials=10)
df_lgbm=lgbm.bestCandidates()
print("LGBM", df_lgbm)