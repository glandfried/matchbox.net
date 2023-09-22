from Matchbox import Matchbox
from RatingPredictors import TrainTestSplitInstance
from Others import LGBM
from Others import RandomForest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#dataset = "./data/MovieLens/ml-20m/ratings.csv"
dataset = "./data/Simulation/simulation_20230922_15-53-20/ratings.csv"
ttsi = TrainTestSplitInstance(dataset)
ttsi.loadDatasets(preprocessed=False, NROWS=None, BATCH_SIZE=None)
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


"""
rf=RandomForest(ttsi, max_trials=100)
df_rf=rf.bestCandidates()
print("RandomForest", df_rf)

lgbm=LGBM(ttsi, max_trials=100)
df_lgbm=lgbm.bestCandidates()c
print("LGBM", df_lgbm)

mbox=Matchbox(ttsi, max_trials=100)
df_mbox=mbox.bestCandidates()
"""

mbox=Matchbox(ttsi, max_trials=1)
params = mbox.bestParams()
params["traitCount"] = 2
params["numLevels"] = 2
dict=mbox.objective(params, return_pred=True)
df_pred = dict["pred"]
df_pred.to_csv(f"./trials/pred_{mbox.resultsName()}.csv", header=True, index=False)
dict.pop("pred")
df = pd.DataFrame.from_dict({k: [v] for k, v in dict.items()})
df["geo_mean"] = np.exp(-df["loss"]) #prediccion promedio, porque si en una productoria de las predicciones reemplazas todas las preds por el valor de la media geometrica, sale este
if "tr_loss" in df.columns:
    df["geo_mean_tr"] = np.exp(-df["tr_loss"])
df["dataset_size"] = mbox.ttsi.size
df.to_csv(f"./trials/{mbox.resultsName()}.csv", header=True, index=False)
print(df)

#df = pd.read_csv("./trials/Matchbox_best_y_simulation.csv")
#y_pred = df["y_pred"]
plt.hist(df_pred["y_pred_proba"])
plt.savefig("./tmp/hist.png")
plt.close()
plt.hist(df_pred["y_pred_proba"], bins=100)
plt.savefig("./tmp/hist_bins.png")
print("Terminado!")
#print("Naive Bayes", RatingPredictors.infer_NaiveBayes(ttsi))
#print("Random Forest", RatingPredictors.infer_RandomForest(ttsi,10))
#OLD#print(RatingPredictors.infer_matchboxnet(dataset, useritem, 900000000).posterior)
#print("Matchbox (infer.NET)", RatingPredictors.infer_matchboxnet(ttsi))
#print(RatingPredictors.infer_SVDpp(dataset, useritem, 900000000).est)

