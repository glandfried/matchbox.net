import InfernetWrapper
from InfernetWrapper import *

def matchbox():
    # Ejemplo de https://dotnet.github.io/infer/userguide/Learners/Matchbox%20recommender/Learner%20API.html
    dataMapping = CsvMapping()
    recommender = Microsoft.ML.Probabilistic.Learners.MatchboxRecommender.Create(dataMapping)
    # Settings: https://dotnet.github.io/infer/userguide/Learners/Matchbox/API/Setting%20up%20a%20recommender.html
    recommender.Settings.Training.TraitCount = 5;
    recommender.Settings.Training.IterationCount = 20;
    recommender.Train("./data/MovieLens/data_50.csv");

    '''
    recommender.Save("TrainedModel.bin");
    // ...
    recommender = MatchboxRecommender.Load<string, string, string, NoFeatureSource>( "TrainedModel.bin");
    '''
    # Modos prediccion: https://dotnet.github.io/infer/userguide/Learners/Matchbox/API/Prediction.html
    posterior = recommender.PredictDistribution("196","302");
    print("Posterior rating para usuario 196 e item 302 con 5 estrellas");
    print(",".join(map(str,posterior)));
    recommendations = recommender.Recommend("196", 10);
    print(",".join(map(str,recommendations)));

    print("Imple dotnet:")
    dotnet_posteriors = {0: 0.12841634145715716, 1: 0.14323634733764679, 2: 0.16840827972422545, 3: 0.2646673070766022, 4: 0.10161728482043446, 5: 0.19365443958393389}
    print(dotnet_posteriors);
    print("Diferencia con implementacion en dotnet:")
    dotnet_ranking= [242,327,234,603,1014,387,95,222,465,201]
    print(["{0:.20f}".format(abs(dotnet_posteriors[i]-posterior[i])) for i in range(len(posterior))])

matchbox()