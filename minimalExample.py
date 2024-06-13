from Matchbox import Matchbox
from RatingPredictors import TrainTestSplitInstance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#dataset = "./data/MovieLens/ml-20m/ratings.csv"
dataset = "./data/Simulation/matchbox_20240116_12-06-00_2traits_400users_lowerNoise/ratings.csv"
ttsi = TrainTestSplitInstance(dataset)
ttsi.loadDatasets(preprocessed=True, NROWS=None, BATCH_SIZE=None)
#ttsi.trainBatches = 8
#ttsi.testBatches = 3
#del ttsi.X_test
#del ttsi.X_train
#print("Loading dataset to DataFrame...")
#dataset = "./data/MovieLens/data_50.csv"

#ttsi.to_csv(ttsi)

#useritem = RatingPredictors.ObservedRating("196","302", 900000000, 3)
#useritem = MatchboxImplementations.ObservedRating("196","302",900000000,1)
#RatingPredictors.infer_matchbox_propio()
#print("Real rating: 3")
#print("LGBM", RatingPredictors.infer_LGBM(ttsi,10))

mbox = Matchbox(ttsi, max_trials=1)
Matchbox.createRecommender()

mbox=Matchbox(ttsi, max_trials=10)
df_mbox=mbox.bestCandidates()
print("Matchbox", df_mbox)

rf=RandomForest(ttsi, max_trials=10)
df_rf=rf.bestCandidates()
print("RandomForest", df_rf)

lgbm=LGBM(ttsi, max_trials=10)
df_lgbm=lgbm.bestCandidates()
print("LGBM", df_lgbm)
