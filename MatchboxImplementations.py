import InfernetWrapper
from InfernetWrapper import *
from tabnanny import verbose
from surprise import SVDpp, Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import split
#import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
#import matchbox_refactor as mbox

class UserItemPair():
    def __init__(self, user, item):
        self.user = user
        self.item = item

class ObservedRating(UserItemPair):
    def __init__(self, user, item, timestamp, value):
        self.value = value
        self.timestamp = timestamp
        super().__init__(user, item)

class EstimatedRating(UserItemPair):
    def __init__(self, user, item, timestamp, estimate):
        self.est = estimate
        self.timestamp = timestamp
        super().__init__(user, item)

def infer_dotnet_propio():
    MatchboxPropio.RecommenderSystem().Run()

def infer_dotnet(datasetName, ui, ts, traitCount=5, iterationCount=20):
    # Ejemplo de https://dotnet.github.io/infer/userguide/Learners/Matchbox%20recommender/Learner%20API.html
    dataMapping = CsvMapping()
    recommender = MatchboxCsvWrapper.Create(dataMapping)
    # Settings: https://dotnet.github.io/infer/userguide/Learners/Matchbox/API/Setting%20up%20a%20recommender.html
    recommender.Settings.Training.TraitCount = traitCount;
    recommender.Settings.Training.IterationCount = iterationCount;
    recommender.Train(datasetName);

    # Modos prediccion: https://dotnet.github.io/infer/userguide/Learners/Matchbox/API/Prediction.html
    #print("Posterior rating para usuario 196 e item 302 con 5 estrellas");
    posterior = recommender.PredictDistribution(ui.user, ui.item);
    predRating = recommender.Predict(ui.user, ui.item);
    #print(",".join(map(str,posterior)));
    #recommendations = recommender.Recommend(rating.user, 10);
    #print(",".join(map(str,recommendations)));
    res = EstimatedRating(ui.user, ui.item, ts, predRating)
    res.posterior = posterior
    return res
    '''
    print("Imple dotnet:")
    dotnet_posteriors = {0: 0.12841634145715716, 1: 0.14323634733764679, 2: 0.16840827972422545, 3: 0.2646673070766022, 4: 0.10161728482043446, 5: 0.19365443958393389}
    print(dotnet_posteriors);
    print("Diferencia con implementacion en dotnet:")
    dotnet_ranking= [242,327,234,603,1014,387,95,222,465,201]
    print(["{0:.20f}".format(abs(dotnet_posteriors[i]-posterior[i])) for i in range(len(posterior))])
    '''

# https://surprise.readthedocs.io/en/stable/getting_started.html#use-a-custom-dataset
def infer_SVDpp(datasetName, ui, ts):
    reader = Reader(line_format="user item rating timestamp", sep=",", rating_scale=(1,5))
    algo = SVDpp()
    data = Dataset.load_from_file(datasetName, reader=reader).build_full_trainset()
    algo.fit(data)

    #trainset, testset = split.train_test_split(data, test_size=0.25)
    #predictions = algo.fit(trainset).test(testset)
    if isinstance(ui, ObservedRating):
        p = algo.predict(ui.user,ui.item, r_ui=ui.value)
    else:
        p = algo.predict(ui.user,ui.item)
    return p

"""
def testRateOneAndPropagate(self):
    plotRatings = False
    h = self.defaultHistory()
    users = self.createUsers(2, 2)
    movies = self.createMovies(2, 2)

    print(f"Rating estimado para usuario {users[1].id} y pelicula {movies[1].id} es: {h.estimateRating(users[1], movies[1],5)}")
    r0_0 = mbox.Rating(users[0], movies[0], 5, plotRatings=plotRatings)
    print(f"Rating estimado para usuario {users[1].id} y pelicula {movies[1].id} es: {h.estimateRating(users[1], movies[1],5)}")
    r0_1 = mbox.Rating(users[0], movies[1], -5, plotRatings=plotRatings)
    print(f"Rating estimado para usuario {users[1].id} y pelicula {movies[1].id} es: {h.estimateRating(users[1], movies[1],5)}")
    r1_0 = mbox.Rating(users[1], movies[0], 5, plotRatings=plotRatings)
    h.addRating(r0_0, t=1)
    h.addRating(r0_1, t=2)
    h.addRating(r1_0, t=3)
    
    h.propagate()
    #h.addRating(r2, 1)
    print(f"Rating estimado para usuario {users[1].id} y pelicula {movies[1].id} es: {h.estimateRating(users[1], movies[1],5)}")

def infer_RandomForest(datasetName, rating, n_estimators=200):
    pd.read_csv(datasetName, )
    rfc = RandomForestClassifier(n_estimators=n_estimators)
    rfc.fit(X_train, y_train)
"""
