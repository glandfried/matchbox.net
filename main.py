import MatchboxImplementations

dataset = "./data/MovieLens/data_50.csv"
useritem = MatchboxImplementations.ObservedRating("196","302", 900000000, 1)
#useritem = MatchboxImplementations.ObservedRating("196","302",900000000,1)
MatchboxImplementations.infer_dotnet_propio()
print(MatchboxImplementations.infer_dotnet(dataset, useritem, 900000000).est)
print(MatchboxImplementations.infer_SVDpp(dataset, useritem, 900000000).est)