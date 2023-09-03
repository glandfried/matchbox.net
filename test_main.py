from Matchbox import Matchbox
from RatingPredictors import TrainTestSplitInstance
from Others import LGBM
import os

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

testDir = "./data/Tests"
tests = [int(x.replace("test", "")) for x in get_immediate_subdirectories(testDir)]
tests.sort()
for i in tests:
#for i in [1,2,4,5]:        
    dataset = f"{testDir}/test{i}/ratings.csv"
    ttsi = TrainTestSplitInstance(dataset)
    ttsi.loadDatasets(preprocessed=True, NROWS=None, BATCH_SIZE=None)

    mbox=Matchbox(ttsi, max_trials=1)
    df_mbox=mbox.predictionResults()
    print(f"==== Test {i} ====")
    print(df_mbox)
