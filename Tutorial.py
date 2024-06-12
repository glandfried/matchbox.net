from Matchbox import Matchbox
from RatingPredictors import TrainTestSplitInstance
import numpy as np
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import binom
import os
import matplotlib.pyplot as plt
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import warnings
from InfernetWrapper import Gaussian
from InfernetWrapper import VariableArray
import sys
import random

if __name__ == "__main__":
    path = f"./data/Simulation/"
    path += datetime.today().strftime('%Y%m%d_%H-%M-%S')
    #path += "simulation_20230926_15-53-20"
    generate_data=True
    readOnly=False

    os.makedirs(path, exist_ok=True)
    if generate_data:
        df, generated_users, generated_items = GenerateData()
    else:
        generated_users = pd.read_csv(f"{path}/user_truth.csv")
        generated_items = pd.read_csv(f"{path}/item_truth.csv")
    
    ttsi = TrainTestSplitInstance(f"{path}/ratings.csv")
    ttsi.loadDatasets(preprocessed=True, NROWS=None, BATCH_SIZE=None)
    mbox=Matchbox(ttsi, max_trials=1)
    params = mbox.bestParams()
    params["iterationCount"] = 1
    params["traitCount"] = numFeatures
    params["minRating"] = 0
    params["maxRating"] = numLevels
    recommender = mbox.createRecommender(params)
    _ = mbox.train(recommender)

    estimated_users, estimated_items = GetPosteriors(recommender, path, readOnly=readOnly)

    dfTest, dfStats = mbox.predictionResults(params, recommender)
    
    if not readOnly:
        dfTest.to_csv(f"{path}/preds.csv", header=True, index=False)
        dfStats.to_csv(f"{path}/stats.csv", header=True, index=False)

    print(dfStats)
    print("Simulación terminada!")
    
    #SimulationPlots(path, generated_users, generated_items, estimated_users, estimated_items)
    res = (generated_users, generated_items, estimated_users, estimated_items)


class RecommenderTutorialFromRepository:
    numUsers: int = 200
    numItems: int = 200
    numTraits: int = 2
    numObs: int = 20000
    numLevels: int = 2

    def dummyRatingsTestSplitFile(self, folder):
        with open(f"{folder}/ratings_test.csv", "w") as file:
            file.write(f"2,2,2,999\n")

    # Generates data from the model
    def GenerateData(
        self, 
        numUsers: int,
        numItems: int,
        numTraits: int,
        numObservations: int,
        numLevels: int,
        userData: VariableArray[int],
        itemData: VariableArray[int],
        ratingData: VariableArray[VariableArray[bool], list[list[bool]]],
        userTraitsPrior: list[list[Gaussian]],
        itemTraitsPrior: list[list[Gaussian]],
        userBiasPrior: list[Gaussian],
        itemBiasPrior: list[Gaussian],
        userThresholdsPrior: list[list[Gaussian]],
        affinityNoiseVariance: float,
        thresholdsNoiseVariance: float,
        printGenerated: bool) -> None:

        generatedUserData = []
        generatedItemData = []
        generatedRatingData = []

        # Sample model parameters from the priors
        random.seed(12347)
        userTraits = [[userTraitsPrior[u][t].Sample() for t in numTraits] for u in numUsers]
        itemTraits = [[itemTraitsPrior[i][t].Sample() for t in numTraits] for u in numItems]
        userBias = [userBiasPrior[u].Sample() for u in numUsers]
        itemBias = [itemBiasPrior[u].Sample() for u in numItems]
        userThresholds = [[userThresholdsPrior[u][l].Sample() for l in numLevels] for u in numUsers]

        # Repeat the model with fixed parameters
        visited = set()
        iObs = 0

        with tqdm(total=numObs) as pbar:
            while iObs < numObs:
                user = random.randrange(numUsers)
                item = random.randrange(numItems)
                userItemPairID = user * numItems + item #pair encoding  

                if userItemPairID in visited: #duplicate generated
                    continue #redo this iteration with different user-item pair

                visited.add(userItemPairID);
                
                products = np.array(userTraits[user]) * np.array(itemTraits[item])
                bias = userBias[user] + itemBias[item]
                affinity = bias + np.sum(products)
                noisyAffinity = norm.rvs(affinity, affinityNoiseVariance)
                noisyThresholds = [norm.rvs(ut, thresholdNoiseVariance) for ut in UserThresholds[user]]

                generatedUserData.append(user);  
                generatedItemData.append(item)
                generatedRatingData.append([1 if noisyAffinity > noisyThresholds[l] else 0 for l in range(len(noisyThresholds))])
                iObs += 1
                pbar.update(1)

        if (printGenerated) :
            print("| true parameters |")
            print("| --------------- |")
            for i in range(5):
                print(f"| {itemTraits[i][0] :.2}    {itemTraits[i][1]:.2} |")

        userData.ObservedValue = generatedUserData;
        itemData.ObservedValue = generatedItemData;
        ratingData.ObservedValue = generatedRatingData;
    
        df = pd.DataFrame.from_dict({"user":generatedUserData, "item":generatedItemData, "ratingList":generatedRatingData})
        df["rating"] = [np.sum(r)-1 for r in generatedRatingData] #Resto 1 porque siempre pasa threshold 0 (TODO: es raro)
        df["timestamps"] = [999 for _ in range(len(generatedRatingData))]

        generated_users.to_csv(f"{path}/user_truth.csv", header=True, index=False)
        generated_items.to_csv(f"{path}/item_truth.csv", header=True, index=False)
        df[["user","item","rating","timestamps"]].to_csv(f"{path}/ratings_train.csv", header=False, index=False)
        self.dummyRatingsTestSplitFile(path)
        return df, generated_users, generated_items


    def Evidence(self, path: str, generateData: bool=True) -> None:
        os.makedirs(path, exist_ok=True)
        if generateData:
            df, generated_users, generated_items = self.GenerateData()
        else:
            generated_users = pd.read_csv(f"{path}/user_truth.csv")
            generated_items = pd.read_csv(f"{path}/item_truth.csv")
        
        ttsi = TrainTestSplitInstance(f"{path}/ratings.csv")
        ttsi.loadDatasets(preprocessed=True, NROWS=None, BATCH_SIZE=None)
        mbox=Matchbox(ttsi, max_trials=1)
        params = mbox.bestParams()
        params["traitCount"] = numFeatures
        recommender = mbox.createRecommender(params)
        _ = mbox.train(recommender)

        estimated_users, estimated_items = GetPosteriors(recommender, path)

        print("Simulación terminada!")
        #SimulationPlots(path, generated_users, generated_items, estimated_users, estimated_items)
        return generated_users, generated_items, estimated_users, estimated_items

        """
        # Define counts
        int numUsers = RecommenderTutorialFromRepository.numUsers;  
        int numItems = RecommenderTutorialFromRepository.numItems;  
        int numTraits = RecommenderTutorialFromRepository.numTraits;  
        Variable<int> numObservations = Variable.Observed(RecommenderTutorialFromRepository.numObs);  
        int numLevels = RecommenderTutorialFromRepository.numLevels;  

        # Define ranges
        Range user = new Range(numUsers).Named("user");
        Range item = new Range(numItems).Named("item");
        Range trait = new Range(numTraits).Named("trait");
        Range observation = new Range(numObservations).Named("observation");
        Range level = new Range(numLevels).Named("level");

        # Define latent variables
        var userTraits = Variable.Array(Variable.Array<double>(trait), user).Named("userTraits");
        var itemTraits = Variable.Array(Variable.Array<double>(trait), item).Named("itemTraits");
        var userBias = Variable.Array<double>(user).Named("userBias");
        var itemBias = Variable.Array<double>(item).Named("itemBias");
        var userThresholds = Variable.Array(Variable.Array<double>(level), user).Named("userThresholds");

        # Define priors
        var userTraitsPrior = Variable.Array(Variable.Array<Gaussian>(trait), user).Named("userTraitsPrior");
        var itemTraitsPrior = Variable.Array(Variable.Array<Gaussian>(trait), item).Named("itemTraitsPrior");
        var userBiasPrior = Variable.Array<Gaussian>(user).Named("userBiasPrior");
        var itemBiasPrior = Variable.Array<Gaussian>(item).Named("itemBiasPrior");
        var userThresholdsPrior = Variable.Array(Variable.Array<Gaussian>(level), user).Named("userThresholdsPrior");

        # Define latent variables statistically
        userTraits[user][trait] = Variable<double>.Random(userTraitsPrior[user][trait]);
        itemTraits[item][trait] = Variable<double>.Random(itemTraitsPrior[item][trait]);
        userBias[user] = Variable<double>.Random(userBiasPrior[user]);
        itemBias[item] = Variable<double>.Random(itemBiasPrior[item]);
        userThresholds[user][level] = Variable<double>.Random(userThresholdsPrior[user][level]);

        # Initialise priors
        Gaussian traitPrior = Gaussian.FromMeanAndVariance(0.0, 1.0);
        Gaussian biasPrior = Gaussian.FromMeanAndVariance(0.0, 1.0);

        userTraitsPrior.ObservedValue = Util.ArrayInit(numUsers, u => Util.ArrayInit(numTraits, t => traitPrior));
        itemTraitsPrior.ObservedValue = Util.ArrayInit(numItems, i => Util.ArrayInit(numTraits, t => traitPrior));
        userBiasPrior.ObservedValue = Util.ArrayInit(numUsers, u => biasPrior);
        itemBiasPrior.ObservedValue = Util.ArrayInit(numItems, i => biasPrior);
        userThresholdsPrior.ObservedValue = Util.ArrayInit(numUsers, u => Util.ArrayInit(numLevels, l => Gaussian.FromMeanAndVariance(l - numLevels / 2.0 + 0.5, 1.0)));

        # Break symmetry and remove ambiguity in the traits
        for (int i = 0; i < numTraits; i++)
        {
            # Assume that numTraits < numItems
            for (int j = 0; j < numTraits; j++)
            {
                itemTraitsPrior.ObservedValue[i][j] = Gaussian.PointMass(0);
            }

            itemTraitsPrior.ObservedValue[i][i] = Gaussian.PointMass(1);
        }

        # Declare training data variables
        var userData = Variable.Array<int>(observation).Named("userData");
        var itemData = Variable.Array<int>(observation).Named("itemData");
        var ratingData = Variable.Array(Variable.Array<bool>(level), observation).Named("ratingData");

        # Set model noises explicitly
        Variable<double> affinityNoiseVariance = Variable.Observed(0.1).Named("affinityNoiseVariance");
        Variable<double> thresholdsNoiseVariance = Variable.Observed(0.1).Named("thresholdsNoiseVariance");

        Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");  
        IfBlock block = Variable.If(evidence); 
        # Model
        using (Variable.ForEach(observation))
        {
            VariableArray<double> products = Variable.Array<double>(trait).Named("products");
            products[trait] = userTraits[userData[observation]][trait] * itemTraits[itemData[observation]][trait];

            Variable<double> bias = (userBias[userData[observation]] + itemBias[itemData[observation]]).Named("bias");
            Variable<double> affinity = (bias + Variable.Sum(products).Named("productSum")).Named("affinity");
            Variable<double> noisyAffinity = Variable.GaussianFromMeanAndVariance(affinity, affinityNoiseVariance).Named("noisyAffinity");

            VariableArray<double> noisyThresholds = Variable.Array<double>(level).Named("noisyThresholds");
            noisyThresholds[level] = Variable.GaussianFromMeanAndVariance(userThresholds[userData[observation]][level], thresholdsNoiseVariance);
            ratingData[observation][level] = noisyAffinity > noisyThresholds[level];
        }
        block.CloseBlock();  

        # This example requires EP
        InferenceEngine engine = new InferenceEngine();
        if (not (engine.Algorithm is Microsoft.ML.Probabilistic.Algorithms.ExpectationPropagation)):
            Console.WriteLine("This example only runs with Expectation Propagation")
            return

        # Observe training data
        GenerateData(
            numUsers,
            numItems,
            numTraits,
            numObservations.ObservedValue,
            numLevels,
            userData,
            itemData,
            ratingData,
            userTraitsPrior.ObservedValue,
            itemTraitsPrior.ObservedValue,
            userBiasPrior.ObservedValue,
            itemBiasPrior.ObservedValue,
            userThresholdsPrior.ObservedValue,
            affinityNoiseVariance.ObservedValue,
            thresholdsNoiseVariance.ObservedValue,
            false);

        # Allow EP to process the product factor as if running VMP
        # as in Stern, Herbrich, Graepel paper.
        engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));
        engine.Compiler.ShowWarnings = true;

        # Run inference
        var userTraitsPosterior = engine.Infer<Gaussian[][]>(userTraits);
        var itemTraitsPosterior = engine.Infer<Gaussian[][]>(itemTraits);
        var userBiasPosterior = engine.Infer<Gaussian[]>(userBias);
        var itemBiasPosterior = engine.Infer<Gaussian[]>(itemBias);
        var userThresholdsPosterior = engine.Infer<Gaussian[][]>(userThresholds);

        # Feed in the inferred posteriors as the new priors
        userTraitsPrior.ObservedValue = userTraitsPosterior;
        itemTraitsPrior.ObservedValue = itemTraitsPosterior;
        userBiasPrior.ObservedValue = userBiasPosterior;
        itemBiasPrior.ObservedValue = itemBiasPosterior;
        userThresholdsPrior.ObservedValue = userThresholdsPosterior;

        double logEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;  
        double modelEvidence = System.Math.Exp(logEvidence);
        Console.WriteLine("\nEvidence:");
        Console.WriteLine("\n|   |   |\n| -------- | - |\n| evidence | {0} |\n| log(evidence) | {1} |\n", modelEvidence, logEvidence.ToString("E2"));
        """