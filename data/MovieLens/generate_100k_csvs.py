print("Loading train and test indexes...")

def load_idxs(path):
    try:
        with open(f"{path}/ratings_train.idx", "r") as f:
            train_idxs = f.read().split(",")
            train_idxs = [int(x) for x in train_idxs]
            train_idxs.sort()
            train_idxs.append(-999)
    except FileNotFoundError:
        train_idxs = [-999]
    try:
        with open(f"{path}/ratings_test.idx", "r") as f:
            test_idxs = f.read().split(",")
            test_idxs = [int(x) for x in test_idxs]
            test_idxs.sort()
            test_idxs.append(-999)
    except FileNotFoundError:
        test_idxs = [-999]
    return train_idxs,test_idxs

def write_header(fs):
    for f in fs:
        f.write("userId,movieId,rating,timestamp")

def write_to_train_or_test(idx, ftrain, ftest, nextTrain, nextTest):
    if idx == nextTrain:
        ftrain.write(",".join(l))
        return (1,0)
    if idx == nextTest:
        ftest.write(",".join(l))
        return (0,1)
    return (0,0)

train_idxs, test_idxs = load_idxs("ml-100k")
train_idxs_bin, test_idxs_bin = load_idxs("ml-100k-binary")
print(f"Processing {len(test_idxs)-1} test ratings and {len(train_idxs)-1} train ratings...")
print("Generating 5 star and binary csvs...")
with open("ml-100k/u.data", "r") as f:
    with (open("ml-100k/ratings.csv", "w") as fw,
        open("ml-100k-binary/ratings.csv", "w") as fw_bin,
        open("ml-100k/ratings_train.csv","w") as fw_train,
        open("ml-100k/ratings_test.csv","w") as fw_test,
        open("ml-100k-binary/ratings_train.csv","w") as fw_train_bin,
        open("ml-100k-binary/ratings_test.csv","w") as fw_test_bin,
        ):
            iline = 0
            itest = 0
            itrain = 0
            itest_bin = 0
            itrain_bin = 0
            write_header([fw,fw_bin])
            for line in f.readlines():
                l = [x for x in line.split("\t")]
                fw.write(",".join(l))

                lint = [int(x) for x in line.split("\t")]
                lint[2] = 1 if lint[2] >= 4 else 0
                lint = [str(x) for x in lint]
                fw_bin.write(",".join(lint))
                
                addTrain, addTest = write_to_train_or_test(iline, fw_train, fw_test, train_idxs[itrain], test_idxs[itest])
                itrain += addTrain
                itest += addTest

                addTrain, addTest = write_to_train_or_test(iline, fw_train_bin, fw_test_bin, train_idxs_bin[itrain_bin], test_idxs_bin[itest_bin])
                itrain_bin += addTrain
                itest_bin += addTest
                
                iline += 1
