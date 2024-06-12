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
from InfernetWrapper import LossFunction
import sys

numUsers = 200
numItems = 200
numFeatures = 2
numLevels = 2
numObs = 20000
affinityNoisePrecision = np.sqrt(0.1)
thresholdNoisePrecision = np.sqrt(0.1)
printAsStrings = True

def GaussSub(g1, g2):
    if (isinstance(g2, int) and g2 == 0):
        return g1
    elif (isinstance(g2, Gaussian)):
        return Gaussian(g1.GetMean()-g2.GetMean(), np.sqrt(g1.GetVariance()+g2.GetVariance()))

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def generateItems(numItems):
    res = []
    for i in range(numItems):
        traits = norm.rvs(0, 1, size=numFeatures)
        res.append(traits)
    return res

def generateThresholds(num):
    res = []
    for _ in range(num):
        userThresholdsPrior = [l - numLevels / 2.0 + 0.5 for l in range(numLevels)]
        thresholds = [norm.rvs(mean,1) for mean in userThresholdsPrior]
        thresholds = [-np.inf] + thresholds + [np.inf]
        #thresholds[3] = 0.0 #Matchbox seems to fix side and middle thresholds in -inf, 0, +inf
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
                        plt.axvline(truth[i], ymin=0 , ymax=1, color=col, linestyle="--", linewidth=1, alpha=0.6)
                    #plt.stem(mean, norm.pdf(mean, mean, np.sqrt(var)), col)
                    # plot means:
                    plt.vlines(mean, ymin=0 , ymax=norm.pdf(mean, mean, np.sqrt(var)), color=col, alpha=0.2)
                    #plt.text(mean, -.05, f'thr_{i}', color=col, ha='center', va='top')

    plt.title(f"User {user} thresholds")
    plt.legend()
    plt.savefig( f"{path}/user{user}_thresholds.png")
    plt.close()

def plotRatingInThresholds(gauss, gauss_pred, rating_data, path="./tmp", truth=None):
    user = rating_data[0] #0 = userId column
    item = rating_data[1] #1 = movieId column
    rating_truth = rating_data[2] #2 = y_test column
    rating_pred = rating_data[3] #3 = y_pred column
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
                        plt.axvline(truth[i], ymin=0 , ymax=1, color=col, linestyle="--", linewidth=1, alpha=0.6)
                    #plt.stem(mean, norm.pdf(mean, mean, np.sqrt(var)), col)
                    # plot means:
                    plt.vlines(mean, ymin=0 , ymax=norm.pdf(mean, mean, np.sqrt(var)), color=col, alpha=0.2)
                    #plt.text(mean, -.05, f'thr_{i}', color=col, ha='center', va='top')

            mean = gauss_pred.GetMean()
            var = gauss_pred.GetVariance()
            p = plt.plot(grilla, norm.pdf(grilla, mean, np.sqrt(var)), '-', color="tab:gray", label=f'r: ({mean:.1f}, {var:.1f}), [truth:{rating_truth}, pred:{rating_pred}]')
            col = p[0].get_color()
            plt.vlines(mean, ymin=0 , ymax=norm.pdf(mean, mean, np.sqrt(var)), color=col, alpha=0.5)

    plt.title(f"User {user}'s rating of item {item}")
    plt.legend()
    plt.savefig( f"{path}/{user}-{item}.png")
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
    itemTraits = generateItems(numItems)
    itemTraits[0][0] = 1
    itemTraits[0][1] = 0
    itemTraits[1][0] = 0
    itemTraits[1][1] = 1
    itemTraits[2][0] = 0.44
    itemTraits[2][1] = -1.10
    itemTraits[3][0] = -0.38
    itemTraits[3][1] = -0.83
    itemTraits[4][0] = 0.11
    itemTraits[4][1] = 0.68
    userTraits = generateItems(numUsers)
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
            if iObs < numFeatures:
                user = iObs
                item = iObs
            userItemPairID = user * numItems + item #pair encoding  
            
            if userItemPairID in visited: #duplicate generated
                continue #redo this iteration with different user-item pair

            visited.add(userItemPairID);
            
            products = np.array(userTraits[user]) * np.array(itemTraits[item])
            bias = userBias[user] + itemBias[item]
            affinity = bias + np.sum(products)
            noisyAffinity = norm.rvs(affinity, affinityNoisePrecision)
            noisyThresholds = [norm.rvs(ut, thresholdNoisePrecision) for ut in UserThresholds[user]]

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

def RomperSimetriaEsImportante():
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
        params["traitCount"] = numFeatures
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

def formatNumToString(var):
    if np.isinf(var):
        var = "np.inf" if var > 0 else "-np.inf"
    else:
        var = f"{var:.5f}"
    return var

def GaussianToString(g):
    if printAsStrings:
        if g.IsPointMass:
            return f"Gaussian.PointMass({formatNumToString(g.Point)})"
        else:
            return f"Gaussian({formatNumToString(g.GetMean())}, {formatNumToString(g.GetVariance())})"  
    else:
        if g.IsPointMass:
            return f"{formatNumToString(g.Point)}"
        else:
            return f"{formatNumToString(g.GetMean())}"  
        
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

def ncsm(g : Gaussian):
    return g.GetMean()**2+g.GetVariance()

def msg2(margS, margT):
    res = []
    for k in range(len(margS)):
        s = margS[k]
        t = margT[k]
        res.append(Gaussian(s.GetMean()*t.GetMean(), (ncsm(s)*ncsm(t)) - ((s.GetMean()**2)*(t.GetMean()**2))))
    return res

def ratingBias(ubias, vbias):
    return Gaussian(ubias.GetMean()+vbias.GetMean(), np.sqrt(ubias.GetVariance() + vbias.GetVariance()))

def ratingEstimate(b_post, user_post, item_post):
    normZ = msg2(user_post, item_post)     
    sumMuZ = sum([zk.GetMean() for zk in normZ])
    sumSigma2Z = sum([zk.GetVariance() for zk in normZ])
    return Gaussian(b_post.GetMean() + sumMuZ, np.sqrt(affinityNoisePrecision**2 + b_post.GetVariance() + sumSigma2Z))

def LoadDatasets(path, generated_users, generated_items, estimated_users, estimated_items):
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
    
    userBiasMask = ["Bias" in col for col in generated_users.columns]
    itemBiasMask = ["Bias" in col for col in generated_items.columns]
    userBiasMask_e = ["Bias" in col for col in estimated_users.columns]
    itemBiasMask_e = ["Bias" in col for col in estimated_items.columns]

    if isinstance(estimated_users.loc[:,userThresholdMask_e].iloc[0,0], str) and "Gaussian" in estimated_users.loc[:,userThresholdMask_e].iloc[0,0]:
        for col in estimated_users.loc[:,userThresholdMask_e].columns:
            estimated_users[col] = [eval(x) for x in estimated_users[col]]
        for col in estimated_users.loc[:,userTraitMask_e].columns:
            estimated_users[col] = [eval(x) for x in estimated_users[col]]
        for col in estimated_items.loc[:,itemTraitMask_e].columns:
            estimated_items[col] = [eval(x) for x in estimated_items[col]]
        for col in estimated_users.loc[:,userBiasMask_e].columns:
            estimated_users[col] = [eval(x) for x in estimated_users[col]]
        for col in estimated_items.loc[:,itemBiasMask_e].columns:
            estimated_items[col] = [eval(x) for x in estimated_items[col]]

    return generated_users, generated_items, estimated_users, estimated_items

def SimulationPlots(path, generated_users=None, generated_items=None, estimated_users=None, estimated_items=None):
    generated_users, generated_items, estimated_users, estimated_items = LoadDatasets(path, generated_users, generated_items, estimated_users, estimated_items)
    os.makedirs(path+"/users", exist_ok=True)
    os.makedirs(path+"/items", exist_ok=True)

    userThresholdMask = ["Threshold" in col for col in generated_users.columns]
    userTraitMask = ["Trait" in col for col in generated_users.columns]
    itemTraitMask = ["Trait" in col for col in generated_items.columns]

    userThresholdMask_e = userThresholdMask + [False]*(estimated_users.shape[1]-generated_users.shape[1])
    userTraitMask_e = userTraitMask + [False]*(estimated_users.shape[1]-generated_users.shape[1])
    itemTraitMask_e = itemTraitMask + [False]*(estimated_items.shape[1]-generated_items.shape[1])

    print("Generating user plots....")
    for user in tqdm(generated_users["User"][:10], total=10): #generated_users.shape[0]):
        plotThresholds(estimated_users.iloc[user][userThresholdMask_e], user, path=path+"/users", truth=generated_users.iloc[int(user)][userThresholdMask])
        plotItemTraits(estimated_users.iloc[user][userTraitMask_e], user, isUser=True, path=path+"/users", truth=generated_users.iloc[user][userTraitMask])
    
    print("Generating item plots....")
    for item in tqdm(generated_items["Item"][:10], total=10): #generated_items.shape[0]):
        plotItemTraits(estimated_items.iloc[item][itemTraitMask_e], item, isUser=False, path=path+"/items", truth=generated_items.iloc[item][itemTraitMask])

def AddMyPred(path, generated_users=None, generated_items=None, estimated_users=None, estimated_items=None, maxPlots=None, file="preds.csv"):
    generated_users, generated_items, estimated_users, estimated_items = LoadDatasets(path, generated_users, generated_items, estimated_users, estimated_items)

    userThresholdMask = ["Threshold" in col for col in generated_users.columns]
    userTraitMask = ["Trait" in col for col in generated_users.columns]
    itemTraitMask = ["Trait" in col for col in generated_items.columns]

    userThresholdMask_e = userThresholdMask + [False]*(estimated_users.shape[1]-generated_users.shape[1])
    userTraitMask_e = userTraitMask + [False]*(estimated_users.shape[1]-generated_users.shape[1])
    itemTraitMask_e = itemTraitMask + [False]*(estimated_items.shape[1]-generated_items.shape[1])

    userBiasMask = ["Bias" in col for col in generated_users.columns]
    itemBiasMask = ["Bias" in col for col in generated_items.columns]
    userBiasMask_e = ["Bias" in col for col in estimated_users.columns]
    itemBiasMask_e = ["Bias" in col for col in estimated_items.columns]

    os.makedirs(path+"/ratings", exist_ok=True)
    df = pd.read_csv(f"{path}/{file}")

    maxPlots = len(df) if maxPlots is None else maxPlots
    for i in tqdm(range(maxPlots), total=maxPlots):
        rating = df.iloc[i].values[:4].astype(int)
        user = rating[0]
        item = rating[1]
        user_post = estimated_users.iloc[user]
        item_post = estimated_items.iloc[item]
        b_post = ratingBias(user_post[userBiasMask_e][0], item_post[itemBiasMask_e][0])
        gauss_pred = ratingEstimate(b_post, item_post[itemTraitMask_e], user_post[userTraitMask_e])
        thr = estimated_users.iloc[user][userThresholdMask_e][1:-1]
        """x
        expected_preds = {}
        for i in range(len(thr)+1):
            if i==0:
                #thi = 1 - GaussProd(gauss_pred, thr[i]).GetProbBetween(0, np.inf)
                thi = GaussSub(gauss_pred, thr[i]).GetProbBetween(-np.inf, 0)
            elif i == len(thr):
                #thi = GaussProd(gauss_pred, thr[i-1]).GetProbBetween(0, np.inf)
                thi = 1-GaussSub(gauss_pred, thr[i-1]).GetProbBetween(-np.inf, 0)
            else:
                #thi = GaussProd(gauss_pred, thr[i-1]).GetProbBetween(0, np.inf) - GaussProd(gauss_pred, thr[i]).GetProbBetween(0, np.inf)
                #thi = GaussSub(gauss_pred, thr[i]).GetProbBetween(-np.inf, 0) - (1 - GaussSub(gauss_pred, thr[i-1]).GetProbBetween(-np.inf, 0))
                thi = 1 - (GaussSub(gauss_pred, thr[i-1]).GetProbBetween(-np.inf, 0) + (1 - GaussSub(gauss_pred, thr[i]).GetProbBetween(-np.inf, 0)))
            
            expected_preds[f"Y_proba_{i}_exp"] = thi
        
        #{'Y_proba_0_exp': 0.8011995759238173, 'Y_proba_1_exp': 0.35253555455746755, 'Y_proba_2_exp': 0.44866402136634975}
        assert(np.sum(expected_preds.values) == 1)
        """
        
        df.loc[i, "expected_pred"] = int(np.sum([gauss_pred>t for t in thr])) #Gaussian comparison is implemented, and is how they do it in recommender tutorial
        df.loc[i, "pred_gauss"] = GaussianToString(gauss_pred)

    df.to_csv(f"{path}/{file}", header=True, index=False)


def GaussMul(norm1, norm2):
    sigma2Star = (1/norm1.GetVariance() + 1/norm2.GetVariance())**(-1) #c*N(mu, sigma) = N(c*mu, c*sigma) #TODO: agregar al pdf
    muStar = norm1.GetMean()/norm1.GetVariance() + norm2.GetMean()/norm2.GetVariance()
    muStar *= sigma2Star
    #c = Gaussian(norm2.mu, np.sqrt(norm1.sigma2+norm2.sigma2)).eval(norm1.mu) #result should be multiplied by c, but is proportional to not multiplying by it
    return Gaussian(muStar, np.sqrt(sigma2Star))

def GaussProd(a:Gaussian , b :Gaussian ):
    return a.op_Multiply(b)
"""
    res = Gaussian()
    if (a.IsPointMass):
        if (b.IsPointMass and not a.Point.Equals(b.Point)):
            raise Exception("All Zero Exception")
        res.set_Point(a.Point)
    elif (b.IsPointMass):
        res.set_Point(b.Point)
    else:
        Precision = a.Precision + b.Precision
        MeanTimesPrecision = a.MeanTimesPrecision + b.MeanTimesPrecision
        if (Precision > sys.float_info.max or abs(MeanTimesPrecision) > sys.float_info.max):
            if (a.IsUniform()): 
                res = b
            elif (b.IsUniform()):
                res = a
            else:
                # (am*ap + bm*bp)/(ap + bp) = am*w + bm*(1-w)
                # w = 1/(1 + bp/ap)
                w = 1 / (1 + b.Precision / a.Precision)
                res.set_Point(a.GetMean() * w + b.GetMean() * (1 - w))
    return res
"""    

def RatingPlots(path, generated_users=None, generated_items=None, estimated_users=None, estimated_items=None, maxPlots=None, file="preds.csv"):
    generated_users, generated_items, estimated_users, estimated_items = LoadDatasets(path, generated_users, generated_items, estimated_users, estimated_items)

    userThresholdMask = ["Threshold" in col for col in generated_users.columns]
    userTraitMask = ["Trait" in col for col in generated_users.columns]
    itemTraitMask = ["Trait" in col for col in generated_items.columns]

    userThresholdMask_e = userThresholdMask + [False]*(estimated_users.shape[1]-generated_users.shape[1])
    userTraitMask_e = userTraitMask + [False]*(estimated_users.shape[1]-generated_users.shape[1])
    itemTraitMask_e = itemTraitMask + [False]*(estimated_items.shape[1]-generated_items.shape[1])

    userBiasMask = ["Bias" in col for col in generated_users.columns]
    itemBiasMask = ["Bias" in col for col in generated_items.columns]
    userBiasMask_e = ["Bias" in col for col in estimated_users.columns]
    itemBiasMask_e = ["Bias" in col for col in estimated_items.columns]

    os.makedirs(path+"/ratings", exist_ok=True)
    df = pd.read_csv(f"{path}/{file}")[["userId","movieId","y_test","y_pred","expected_pred","pred_gauss"]]
    #df = df[(df["y_pred"] == df["y_test"]) & (df["y_pred"]==1)]
    save_gauss = "pred_gauss" in df.columns

    maxPlots = len(df) if maxPlots is None else maxPlots
    for i in tqdm(range(maxPlots), total=maxPlots):
        rating_data = df.iloc[i][:-1].values.astype(int)
        user = rating_data[0]
        gauss_pred = eval( df.iloc[i]["pred_gauss"] )
        plotRatingInThresholds(estimated_users.iloc[user][userThresholdMask_e], rating_data=rating_data, gauss_pred=gauss_pred, path=path+"/ratings", truth=generated_users.iloc[int(user)][userThresholdMask])
    
def RatingStats(path, file="preds.csv"):
    df = pd.read_csv(f"{path}/{file}")[["userId","movieId","y_test","y_pred","expected_pred","pred_gauss"]]
    missed_matchbox = len(df[df["y_pred"] != df["y_test"]])
    missed_mine = len(df[df["expected_pred"] != df["y_test"]])
    pred_mismatchs = len(df[df["y_pred"] != df["expected_pred"]])
    print(f"From {len(df)} ratings, there were {pred_mismatchs} prediction mismatches between Matchbox (1 iteration) and our calculations.")
    print(f"Matchbox got {len(df)-missed_matchbox} right and {missed_matchbox} wrong ratings ({100-int(missed_matchbox*100/len(df))}% hit rate).")
    print(f"We got {len(df)-missed_mine} right and {missed_mine} wrong ratings ({100-int(missed_mine*100/len(df))}% hit rate).")

def GetPosteriors(recommender, path, readOnly=False):
    userPosteriors = recommender.GetPosteriorDistributions().Users
    itemPosteriors = recommender.GetPosteriorDistributions().Items

    estimated_users = pd.DataFrame.from_dict({"user": [int(x) for x in userPosteriors.Keys]}).sort_values("user").reset_index(drop=True)
    estimated_items = pd.DataFrame.from_dict({"item": [int(x) for x in itemPosteriors.Keys]}).sort_values("item").reset_index(drop=True)

    print("Generating user data...")
    for user in tqdm(userPosteriors.Keys, total=len(userPosteriors.Keys)):
        posteriors = userPosteriors.get_Item(user)
        #os.makedirs(path+"/users", exist_ok=True)
        #plotThresholds(posteriors.Thresholds, user, path=path+"/users", truth=generated_users.iloc[int(user)][userThresholdMask])
        #plotItemTraits(posteriors.Traits, user, isUser=True, path=path+"/users", truth=generated_users.iloc[int(user)][userTraitMask])
        estimated_users = addEstimatedValues(estimated_users, int(user), thr=posteriors.Thresholds, tra=posteriors.Traits, bias=posteriors.Bias)

    print("Generating item data...")
    for item in tqdm(itemPosteriors.Keys, total=len(itemPosteriors.Keys)):
        posteriors = itemPosteriors.get_Item(item)
        #os.makedirs(path+"/items", exist_ok=True)
        #plotItemTraits(posteriors.Traits, item, isUser=False, path=path+"/items", truth=generated_items.iloc[int(item)][itemTraitMask])
        estimated_items = addEstimatedValues(estimated_items, int(item), tra=posteriors.Traits, bias=posteriors.Bias)
    
    estimated_users.dropna(inplace=True)
    estimated_items.dropna(inplace=True)
    if not readOnly:
        estimated_users.to_csv(f"{path}/user_estimated.csv", header=True, index=False)
        estimated_items.to_csv(f"{path}/item_estimated.csv", header=True, index=False)

    return estimated_users, estimated_items

def Simulation(path, generate_data=True, readOnly=False):
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
    return generated_users, generated_items, estimated_users, estimated_items

def Today():
    return datetime.today().strftime('%Y%m%d_%H-%M-%S')

if __name__ == "__main__":
    path = f"./data/Simulation/"
    """
    path += Today()
    generated_users, generated_items, estimated_users, estimated_items = Simulation(path, generate_data=True)
    SimulationPlots(path, generated_users, generated_items, estimated_users, estimated_items)
    
    path += "20230922_15-19-33"
    SimulationPlots(path)
    """
    path += "simulation_20230926_15-53-20"
    generated_users, generated_items, estimated_users, estimated_items = Simulation(path, generate_data=False, readOnly=False)
    AddMyPred(path, file="preds.csv")
    RatingStats(path, file="preds.csv")

    #RatingPlots(path, maxPlots=10)