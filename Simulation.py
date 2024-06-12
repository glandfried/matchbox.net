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

largeData = True

numItems = 400 if largeData else 10 
numUsers = 400 if largeData else 50
numThresholds = 2
numTraits = 2
numObs = int(numUsers * numItems / 2)
budget = numTraits

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def generateItems(numItems, numTraits, a=0.5, b=0.5):
    res = []
    for i in range(numItems):
        """
        ls = beta.rvs(a, b, size=numTraits)
        total_percentages = ((ls/np.sum(ls)))
        traits = total_percentages * numTraits #Why do this?
        traits *= np.array([-1 if x==1 else 1 for x in binom.rvs(n=1, p=0.5, size=numTraits)])
        """
        traits = norm.rvs(loc=0, scale=1, size=numTraits)
        res.append(traits)
    return res

def generateThresholds(num):
    #assert numThresholds==5, "Only 5 user thresholds supported currently, check comments to implement other quantities"
    res = []
    for _ in range(num):
        #baseThresholds = [-2.55, -1.16, 0.0, 1.19, 2.69] 
        baseThresholds = [Gaussian.FromMeanAndVariance(l - numThresholds / 2.0 + 0.5, 1.0).GetMean() for l in range(numThresholds)]
        noise = [x + norm.rvs(0,0.044) for x in baseThresholds]
        thresholds = [-np.inf] + noise + [np.inf]
        if numThresholds>2 and numThresholds%2 != 0:
            thresholds[int(numThresholds/2)] = 0.0 #Matchbox seems to fix side and middle thresholds in -inf, 0, +inf
        assert([thresholds[i] > thresholds[i-1]+0.1 for i in range(1, len(thresholds)-1)])
        res.append(thresholds)
    return res

def plotGaussian(mean, var, name, path="./tmp"):
    grilla = list(np.arange(0,1,0.01))
    plt.plot(grilla, norm.pdf(grilla, mean, np.sqrt(var)), '-', title=f'{name} - mean:{mean}, var:{var}')
    plt.savefig(f"{path}/{name}.png")
    plt.close()

def plotThresholds(gauss, user, path="./tmp", truth=None):
    grilla = list(np.arange(-5,5,0.1))
    with np.errstate(divide='ignore', invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, g in zip(range(len(gauss)), gauss):
                mean = g.GetMean()
                var = g.GetVariance()
                p = plt.plot(grilla, norm.pdf(grilla, mean, np.sqrt(var)), '-', label=f'{i}: ({mean:.1f}, {var:.1f}){f" [{truth[i]:.1f}]" if truth is not None else ""}')
                col = p[0].get_color()
                if not np.isinf(mean):
                    if var == 0:
                        # If variance is 0 (middle threshold), print line in x=mean from bottom to top of graph
                        plt.axvline(mean, ymin=0 , ymax=1, color=col, alpha=0.5)
                    if truth is not None:
                        # If truth was provided, plot it for this threshold ina  grey dotted line
                        plt.axvline(truth[i], ymin=0 , ymax=1, color="tab:gray", linestyle="--", alpha=0.4)
                    #plt.stem(mean, norm.pdf(mean, mean, np.sqrt(var)), col)
                    plt.vlines(mean, ymin=0 , ymax=norm.pdf(mean, mean, np.sqrt(var)), color=col, alpha=0.5)
                    #plt.text(mean, -.05, f'thr_{i}', color=col, ha='center', va='top')

    plt.title(f"User {user} thresholds")
    plt.legend()
    plt.savefig( f"{path}/user{user}_thresholds.png")
    plt.close()

def plotItemTraits(gauss, item, isUser=False, path="./tmp", truth=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Create subplots 
        #fig, axes = plt.subplots(nrows=3, ncols=2)
        #fig.subplots_adjust(hspace=1)
        #fig.suptitle(f'Traits of {"item" if not isUser else "user"} {item}')
        #fig.set_figheight(8)
        #fig.set_figwidth(7)

        grilla = list(np.arange(-4,4,0.01))
        """
        # Generate dist plots
        for i, ax, g in zip(range(len(gauss)), axes.flatten(), gauss):
            mean = g.GetMean()
            var = g.GetVariance()
            ax.plot(grilla, norm.pdf(grilla, mean, np.sqrt(var)), '-', label=f'{i}: ({mean:.1f}, {var:.1f}){f" [{truth[i]:.1f}]" if truth is not None else ""}')
            ax.legend()
            if truth is not None:
                # If truth was provided, plot it for this threshold ina  grey dotted line
                ax.axvline(truth[i], ymin=0 , ymax=1, color="tab:gray", linestyle="--", alpha=0.4)
            ax.set_title(f"Trait {i}")
        """
        for i, g in zip(range(len(gauss)), gauss):
            mean = g.GetMean()
            var = g.GetVariance()
            p = plt.plot(grilla, norm.pdf(grilla, mean, np.sqrt(var)), '-', label=f'{i}: ({mean:.2f}, {var:.2f}){f" [{truth[i]:.2f}]" if truth is not None else ""}')
            col = p[0].get_color()
            if not np.isinf(mean):
                #if var == 0:
                    # If variance is 0 (middle threshold), print line in x=mean from bottom to top of graph
                    #plt.axvline(mean, ymin=0 , ymax=1, color=col, alpha=0.5)
                if truth is not None:
                    # If truth was provided, plot it for this threshold ina  grey dotted line
                    plt.axvline(truth[i], ymin=0 , ymax=1, color=col, linewidth=1, alpha=0.6)
                #plt.stem(mean, norm.pdf(mean, mean, np.sqrt(var)), col)
                #plt.vlines(mean, ymin=0 , ymax=norm.pdf(mean, mean, np.sqrt(var)), color=col, alpha=0.5)
                #plt.text(mean, -.05, f'thr_{i}', color=col, ha='center', va='top')

    plt.title(f'Traits of {"item" if not isUser else "user"} {item}')
    plt.legend()
    plt.savefig( f"{path}/{'item' if not isUser else 'user'}{item}_traits.png")
    plt.close()

def dummyRatingsTestSplitFile(folder):
    with open(f"{folder}/ratings_test.csv", "w") as file:
      file.write(f"2,2,2,999\n")

def GenerateData():
    affinityNoiseVariance = 0.044
    thresholdNoiseVariance = 0.044
    itemTraits = generateItems(numItems, numTraits)
    #for i in range(numTraits): # Break symmetry
    #    itemTraits[i][0:i] = 0
    #    itemTraits[i][i] = 1
    #    itemTraits[i][i+1:] = 0
    userTraits = generateItems(numUsers, numTraits)
    itemBias = norm.rvs(0,1,size=numItems)
    userBias = norm.rvs(0,1,size=numUsers)
    UserThresholds = generateThresholds(numUsers)

    generated_users = pd.DataFrame.from_dict({"User":list(range(numUsers))})
    for user in range(numUsers):
        generated_users = addEstimatedValues(generated_users, user, thr=UserThresholds[user], tra=userTraits[user])
    generated_users["Bias"] = userBias

    generated_items = pd.DataFrame.from_dict({"Item":list(range(numItems))})
    for item in range(numItems):
        generated_items = addEstimatedValues(generated_items, item, thr=None, tra=itemTraits[item])
    generated_items["Bias"] = itemBias

    generatedUserData = []
    generatedItemData = []
    generatedRatingData = []
    
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

    df = pd.DataFrame.from_dict({"user":generatedUserData, "item":generatedItemData, "ratingList":generatedRatingData})
    df["rating"] = [np.sum(r)-1 for r in generatedRatingData] #Resto 1 porque siempre pasa threshold 0 (TODO: es raro)
    df["timestamps"] = [999 for _ in range(len(generatedRatingData))]

    generated_users.to_csv(f"{path}/user_truth.csv", header=True, index=False)
    generated_items.to_csv(f"{path}/item_truth.csv", header=True, index=False)
    df[["user","item","rating","timestamps"]].to_csv(f"{path}/ratings_train.csv", header=False, index=False)
    dummyRatingsTestSplitFile(path)

    return df, generated_users, generated_items

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

def SimulationPlots(path, generated_users=None, generated_items=None, estimated_users=None, estimated_items=None):
    generated_users = generated_users if generated_users is not None else pd.read_csv(f"{path}/user_truth.csv")
    generated_items = generated_items if generated_items is not None else pd.read_csv(f"{path}/item_truth.csv")
    estimated_users = estimated_users if estimated_users is not None else pd.read_csv(f"{path}/user_estimated.csv")
    estimated_items = estimated_items if estimated_items is not None else pd.read_csv(f"{path}/item_estimated.csv")
    
    userThresholdMask = ["Threshold" in col for col in generated_users.columns]
    userTraitMask = ["Trait" in col for col in generated_users.columns]
    itemTraitMask = ["Trait" in col for col in generated_items.columns]

    userThresholdMask_e = userThresholdMask + [False]*(estimated_users.shape[1]-generated_users.shape[1])
    userTraitMask_e = userTraitMask + [False]*(estimated_users.shape[1]-generated_users.shape[1])
    itemTraitMask_e = itemTraitMask + [False]*(estimated_items.shape[1]-generated_items.shape[1])
    
    if "Gaussian" in estimated_users.loc[:,userThresholdMask_e].iloc[0,0]:
        for col in estimated_users.loc[:,userThresholdMask_e].columns:
            estimated_users[col] = [eval(x) for x in estimated_users[col]]
        for col in estimated_users.loc[:,userTraitMask_e].columns:
            estimated_users[col] = [eval(x) for x in estimated_users[col]]
        for col in estimated_items.loc[:,itemTraitMask_e].columns:
            estimated_items[col] = [eval(x) for x in estimated_items[col]]

    os.makedirs(path+"/users", exist_ok=True)
    os.makedirs(path+"/items", exist_ok=True)
    
    print("Generating user plots....")
    for user in tqdm(generated_users["User"][:10], total=10): #generated_users.shape[0]):
        plotThresholds(estimated_users.iloc[user][userThresholdMask_e], user, path=path+"/users", truth=generated_users.iloc[int(user)][userThresholdMask])
        plotItemTraits(estimated_users.iloc[user][userTraitMask_e], user, isUser=True, path=path+"/users", truth=generated_users.iloc[user][userTraitMask])
    
    print("Generating item plots....")
    for item in tqdm(generated_items["Item"][:10], total=10): #generated_items.shape[0]):
        plotItemTraits(estimated_items.iloc[item][itemTraitMask_e], item, isUser=False, path=path+"/items", truth=generated_items.iloc[item][itemTraitMask])

def GetPosteriors(recommender, path):
    userPosteriors = recommender.GetPosteriorDistributions().Users
    itemPosteriors = recommender.GetPosteriorDistributions().Items

    estimated_users = pd.DataFrame.from_dict({"user": [int(x) for x in userPosteriors.Keys]}).sort_values("user").reset_index(drop=True)
    estimated_items = pd.DataFrame.from_dict({"item": [int(x) for x in itemPosteriors.Keys]}).sort_values("item").reset_index(drop=True)

    print("Getting user posteriors...")
    for user in tqdm(userPosteriors.Keys, total=len(userPosteriors.Keys)):
        posteriors = userPosteriors.get_Item(user)
        #os.makedirs(path+"/users", exist_ok=True)
        #plotThresholds(posteriors.Thresholds, user, path=path+"/users", truth=generated_users.iloc[int(user)][userThresholdMask])
        #plotItemTraits(posteriors.Traits, user, isUser=True, path=path+"/users", truth=generated_users.iloc[int(user)][userTraitMask])
        estimated_users = addEstimatedValues(estimated_users, int(user), thr=posteriors.Thresholds, tra=posteriors.Traits, bias=posteriors.Bias)

    print("Getting item posteriors...")
    for item in tqdm(itemPosteriors.Keys, total=len(itemPosteriors.Keys)):
        posteriors = itemPosteriors.get_Item(item)
        #os.makedirs(path+"/items", exist_ok=True)
        #plotItemTraits(posteriors.Traits, item, isUser=False, path=path+"/items", truth=generated_items.iloc[int(item)][itemTraitMask])
        estimated_items = addEstimatedValues(estimated_items, int(item), tra=posteriors.Traits, bias=posteriors.Bias)
    
    estimated_users.to_csv(f"{path}/user_estimated.csv", header=True, index=False)
    estimated_items.to_csv(f"{path}/item_estimated.csv", header=True, index=False)

    return estimated_users, estimated_items

def Simulation(path, generate_data=True):
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
    params["minRating"] = 0
    params["maxRating"] = numThresholds
    params["traitCount"] = numTraits
    recommender = mbox.createRecommender(params)

    print("Training...")
    stats = mbox.objective(params, recommender=recommender)
    df = pd.DataFrame.from_dict(mbox._formatOutput([{"result":stats}]))
    stats = mbox._addOtherStats(df)
    stats.pop("loss") #Only train dataset, no test data
    stats.pop("rmse") #Only train dataset, no test data
    stats.pop("geo_mean") #Only train dataset, no test data
    print(stats)
    stats.to_csv(f"{path}/results.csv", header=True, index=True)

    estimated_users, estimated_items = GetPosteriors(recommender, path)

    print("Simulación terminada!")
    #SimulationPlots(path, generated_users, generated_items, estimated_users, estimated_items)
    return generated_users, generated_items, estimated_users, estimated_items

if __name__ == "__main__":
    path = f"./data/Simulation/"
    path += datetime.today().strftime('%Y%m%d_%H-%M-%S')
    #path += "mbox_train_20230915"
    generated_users, generated_items, estimated_users, estimated_items = Simulation(path, generate_data=True)
    #SimulationPlots(path)
    SimulationPlots(path, generated_users, generated_items, estimated_users, estimated_items)