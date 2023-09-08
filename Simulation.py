from Matchbox import Matchbox
from RatingPredictors import TrainTestSplitInstance
import numpy as np
from scipy.stats import beta
from scipy.stats import norm
import os
import matplotlib.pyplot as plt

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def generateItems(numItems, numTraits, a=0.5, b=0.5):
    grilla = list(np.arange(0,1,0.01))

    #plt.plot(grilla, beta.pdf(grilla, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')
    #plt.savefig("./tmp/beta.png")
    #plt.clf()

    res = []
    for i in range(100):
        ls = beta.rvs(a, b, size=5)
        res.append((ls/np.sum(ls))*100)

    res = np.array(res)
    #print(np.matrix(res))

def plotGaussian(mean, var, name):
    grilla = list(np.arange(0,1,0.01))
    plt.plot(grilla, norm.pdf(grilla, mean, var), '-', title=f'{name} - mean:{mean}, var:{var}')
    plt.savefig(f"./tmp/{name}.png")
    plt.clf()

def plotThresholds(gauss, user):
    grilla = list(np.arange(-6,6,0.1))
    with np.errstate(divide='ignore', invalid="ignore"):
        for i, g in zip(range(len(gauss)), gauss):
            mean = g.GetMean()
            var = g.GetVariance()
            p = plt.plot(grilla, norm.pdf(grilla, mean, var), '-', label=f'{i}: ({mean:.1f}, {var:.1f})')
            col = p[0].get_color()
            ax = plt.gca()
            if not np.isinf(mean):
                #plt.stem(mean, norm.pdf(mean, mean, var), col)
                plt.vlines(mean, ymin=0 , ymax=norm.pdf(mean, mean, var), color=col, alpha=0.5)
                #plt.text(mean, -.05, f'thr_{i}', color=col, ha='center', va='top')

    plt.title(f"User {user} thresholds")
    plt.legend()
    plt.savefig( f"./tmp/user{user}_thresholds.png")
    plt.clf()

class UserThresholds():
    balancedRating = [(-np.inf, 0.00), (-2.49, 0.33), (-1.12, 0.24), (0.00, 0.00), (1.12, 0.24), (2.47, 0.32), (np.inf, 0.00)]
    
if __name__ == "__main__":
    movies = generateItems(100, 5)
    people = generateItems(10, 5)

    testDir = "./data/Tests"
    tests = [int(x.replace("test", "")) for x in get_immediate_subdirectories(testDir)]
    tests.sort()
    for i in tests[-1:]:
        print(F"========== TEST {i} ==========")
        dataset = f"{testDir}/test{i}/ratings.csv"
        ttsi = TrainTestSplitInstance(dataset)
        ttsi.loadDatasets(preprocessed=True, NROWS=None, BATCH_SIZE=None)
        mbox=Matchbox(ttsi, max_trials=1)
        recommender = mbox.createRecommender(mbox.bestParams())
        _ = mbox.train(recommender)

        userPosteriors = recommender.GetPosteriorDistributions().Users
        itemPosteriors = recommender.GetPosteriorDistributions().Items

        for user in userPosteriors.Keys:
            posteriors = userPosteriors.get_Item(user)
            #os.makedirs(f"./tmp/user{user}", exist_ok=True)
            plotThresholds(posteriors.Thresholds, user)
            print([(float("{:.2f}".format(g.GetMean())), float("{:.2f}".format(g.GetVariance()))) for g in posteriors.Thresholds])
            #for i, thr in zip(range(len(posteriors.Thresholds)), posteriors.Thresholds):
            #    plotGaussian(thr.GetMean(), thr.GetVariance(), f"user{user}/thr-{i}")
        input("Press Enter to continue...") 

    print("Terminado!")