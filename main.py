import RatingPredictors

dataset = "./data/MovieLens/data_50.csv"
ttsi = RatingPredictors.TrainTestSplitInstance(dataset)
#ttsi.to_csv(ttsi)

useritem = RatingPredictors.ObservedRating("196","302", 900000000, 3)
#useritem = MatchboxImplementations.ObservedRating("196","302",900000000,1)
#RatingPredictors.infer_matchbox_propio()
print("Real rating: 3")
#print("Naive Bayes", RatingPredictors.infer_NaiveBayes(ttsi))
print("Random Forest", RatingPredictors.infer_RandomForest(ttsi,10))
print("LGBM", RatingPredictors.infer_LGBM(ttsi,10))
#OLD#print(RatingPredictors.infer_matchboxnet(dataset, useritem, 900000000).posterior)
print("Matchbox (infer.NET)", RatingPredictors.infer_matchboxnet(ttsi))
#print(RatingPredictors.infer_SVDpp(dataset, useritem, 900000000).est)

