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

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def generateItems(numItems, numTraits, a=0.5, b=0.5):
    res = []
    for i in range(numItems):
        ls = beta.rvs(a, b, size=5)
        total_percentages = ((ls/np.sum(ls)))
        traits = total_percentages * numTraits
        traits *= np.array([-1 if x==1 else 1 for x in binom.rvs(n=1, p=0.5, size=5)])
        res.append(traits)

    return res

def generateThresholds(num):
    res = []
    for _ in range(num):
        baseThresholds = [-2.55, -1.16, 0.0, 1.19, 2.69]
        noise = [x + norm.rvs(0,0.44) for x in baseThresholds]
        thresholds = [-np.inf] + noise + [np.inf]
        assert([thresholds[i] > thresholds[i-1]+0.1 for i in range(1, len(thresholds)-1)])
        res.append(thresholds)
    return res

def plotGaussian(mean, var, name):
    grilla = list(np.arange(0,1,0.01))
    plt.plot(grilla, norm.pdf(grilla, mean, var), '-', title=f'{name} - mean:{mean}, var:{var}')
    plt.savefig(f"./tmp/{name}.png")
    plt.close()

def plotThresholds(gauss, user):
    grilla = list(np.arange(-5,5,0.1))
    with np.errstate(divide='ignore', invalid="ignore"):
        for i, g in zip(range(len(gauss)), gauss):
            mean = g.GetMean()
            var = g.GetVariance()
            p = plt.plot(grilla, norm.pdf(grilla, mean, var), '-', label=f'{i}: ({mean:.1f}, {var:.1f})')
            col = p[0].get_color()
            ax = plt.gca()
            if not np.isinf(mean):
                if var == 0:
                    plt.axvline(mean, ymin=0 , ymax=1, color=col, alpha=0.5)    
                #plt.stem(mean, norm.pdf(mean, mean, var), col)
                plt.vlines(mean, ymin=0 , ymax=norm.pdf(mean, mean, var), color=col, alpha=0.5)
                #plt.text(mean, -.05, f'thr_{i}', color=col, ha='center', va='top')

    plt.title(f"User {user} thresholds")
    plt.legend()
    plt.savefig( f"./tmp/user{user}_thresholds.png")
    plt.close()

def plotItemTraits(gauss, item, isUser=False):
    N = 100

    # Create subplots 
    fig, axes = plt.subplots(nrows=3, ncols=2)
    fig.subplots_adjust(hspace=1)
    fig.suptitle(f'Traits of {"item" if not isUser else "user"} {item}')
    fig.set_figheight(8)
    fig.set_figwidth(7)

    # Generate dist plots
    grilla = list(np.arange(-1,1,0.01))
    for i, ax, g in zip(range(len(gauss)), axes.flatten(), gauss):
        mean = g.GetMean()
        var = g.GetVariance()
        ax.plot(grilla, norm.pdf(grilla, mean, var), '-', label=f'{i}: ({mean:.1f}, {var:.1f})')
        ax.legend()
        ax.set_title(f"Trait {i}")
    
    plt.savefig( f"./tmp/{'item' if not isUser else 'user'}{item}_traits.png")
    plt.close()

class UserThresholds():
    balancedRating = [(-np.inf, 0.0), (-2.55, 0.32), (-1.16, 0.24), (0.0, 0.0), (1.19, 0.25), (2.69, 0.39), (np.inf, 0.0)]

def Simulation():
    numItems = 100
    numUsers = 10
    numFeatures = 5
    numObs = 50
    affinityNoiseVariance = 0.44
    thresholdNoiseVariance = 0.44
    itemTraits = generateItems(numItems, numFeatures)
    userTraits = generateItems(numUsers, numFeatures)
    itemBias = norm.rvs(0,1,size=numItems)  
    userBias = norm.rvs(0,1,size=numUsers)  
    UserThresholds = generateThresholds(numUsers)

    generatedUserData = []
    generatedItemData = []
    generatedRatingData = []
    
    visited = set()
    iObs = 0

    with tqdm(total=100) as pbar:
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
            pbar.update(iObs)

    df = pd.DataFrame.from_dict({"user":generatedUserData, "item":generatedItemData, "ratingList":generatedRatingData})
    df["rating"] = [np.sum(r) for r in generatedRatingData]
    return df

def RomperSimetriaEsImportante():
    movies = generateItems(100, 5)
    people = generateItems(10, 5)

    testDir = "./data/Tests"
    tests = [int(x.replace("test", "")) for x in get_immediate_subdirectories(testDir)]
    tests.sort()
    print("Para ver el efecto abrir los graficos de user traits e items traits y mirarlos despues de correr con los datos del test 8 (rompe simetria usando items que despues otros usuarios púntuan), y despues de correr con los datos del test 10 (rompe simetria usando items que despues nadie mas puntua).")
    for i in [8,10]:
        print(F"========== TEST {i} ==========")
        dataset = f"{testDir}/test{i}/ratings.csv"
        ttsi = TrainTestSplitInstance(dataset)
        ttsi.loadDatasets(preprocessed=True, NROWS=None, BATCH_SIZE=None)
        mbox=Matchbox(ttsi, max_trials=1)
        params = mbox.bestParams()
        params["traitCount"] = 5
        recommender = mbox.createRecommender(params)
        _ = mbox.train(recommender)

        userPosteriors = recommender.GetPosteriorDistributions().Users
        itemPosteriors = recommender.GetPosteriorDistributions().Items
        
        for user in userPosteriors.Keys:
            posteriors = userPosteriors.get_Item(user)
            #os.makedirs(f"./tmp/user{user}", exist_ok=True)
            plotThresholds(posteriors.Thresholds, user)
            plotItemTraits(posteriors.Traits, user, isUser=True)
            #print([(float("{:.2f}".format(g.GetMean())), float("{:.2f}".format(g.GetVariance()))) for g in posteriors.Thresholds])  

        for item in itemPosteriors.Keys:
            posteriors = itemPosteriors.get_Item(item)
            #os.makedirs(f"./tmp/item{item}", exist_ok=True)
            #plotThresholds(posteriors.Thresholds, item)
            print(f"ITEM {item}")
            print([(float("{:.2f}".format(g.GetMean())), float("{:.2f}".format(g.GetVariance()))) for g in posteriors.Traits])
            plotItemTraits(posteriors.Traits, item)

        input("Press Enter to continue...") 

    print("Terminado!")

if __name__ == "__main__":
    df = Simulation()
    #df[["user","item","rating"]].to_csv(header=False)
    print(df)