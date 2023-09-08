import os

# TODO: check tests 1-5 and run, add tests 6, 7
MAX_RATING = 5
MIN_RATING = 0

def rate(file, users, movies, rating):
  for u in users:
    for m in movies:
      file.write(f"{u},{m},{rating},999\n")

def test1(folder="data/Tests/test1"):
  os.makedirs(folder, exist_ok=True)
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, [0], [1,2,3,4,5], MIN_RATING)
    rate(file, file, [1,2,3,4,5], [0], MAX_RATING)
  with open(f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING)

def test2(folder="data/Tests/test2"):
  os.makedirs(folder, exist_ok=True)
  test1(folder)
  with open(f"{folder}/ratings_train.csv", "w+") as file:
    rate(file, [1], [1,2,3,4,5], MIN_RATING)

def test3(folder="data/Tests/test3"):
  os.makedirs(folder, exist_ok=True)
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, [0], [1,2,3,4,5], MAX_RATING)
    rate(file, file, [1,2,3,4,5], [0], MAX_RATING)
  with open(f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING)

def test_med(folder, med_rating):
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, [0], [1,2,3,4,5], med_rating)
    rate(file, [1], [1,2,3,4,5], med_rating)
    rate(file, file, [1,2,3,4,5], [0], MAX_RATING)
  with open(f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING)

def test4(folder="data/Tests/test4"):
  os.makedirs(folder, exist_ok=True)
  test_med(3)

def test5(folder="data/Tests/test5"):
  os.makedirs(folder, exist_ok=True)
  test_med(2)

def test8(folder="data/Tests/test8"):
  os.makedirs(folder, exist_ok=True)
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, list(range(0,6)), [0,6,12], 0)
    rate(file, list(range(0,6)), [1,7,13], 1)
    rate(file, list(range(0,6)), [2,8,14], 2)
    rate(file, list(range(0,6)), [3,9,15], 3)
    rate(file, list(range(0,6)), [4,10,16], 4)
    rate(file, list(range(0,6)), [5,11,17], 5)
  with open(f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING)

def generar_tests_csvs():
  test1()
  test2()
  test3()
  test4()
  test5()
  
test8()
#generar_tests_csvs()