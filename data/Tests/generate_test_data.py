import os

# TODO: check tests 1-5 and run, add tests 6, 7
MAX_RATING = 5
MIN_RATING = 0

def rate(file, users, movies, rating, ts=999):
  for u in users:
    for m in movies:
      file.write(f"{u},{m},{rating},{ts}\n")

def test1(folder="data/Tests/test1"):
  os.makedirs(folder, exist_ok=True)
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, [0], [1,2,3,4,5], MIN_RATING)
    rate(file, [1,2,3,4,5], [0], MAX_RATING)
  with open(f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING, 1000)

def test2(folder="data/Tests/test2"):
  os.makedirs(folder, exist_ok=True)
  test1(folder)
  with open(f"{folder}/ratings_train.csv", "a") as file:
    rate(file, [1], [1,2,3,4,5], MIN_RATING)

def test3(folder="data/Tests/test3"):
  os.makedirs(folder, exist_ok=True)
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, [0], [1,2,3,4,5], MAX_RATING)
    rate(file, [1,2,3,4,5], [0], MAX_RATING)
  with open(f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING, 1000)

def test_med(med_rating, folder):
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, [0], [1,2,3,4,5], med_rating)
    rate(file, [1], [1,2,3,4,5], med_rating)
    rate(file, [1,2,3,4,5], [0], MAX_RATING)
  with open(f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING, 1000)

def test4(folder="data/Tests/test4"):
  os.makedirs(folder, exist_ok=True)
  test_med(3, folder)

def test5(folder="data/Tests/test5"):
  os.makedirs(folder, exist_ok=True)
  test_med(2, folder)

def test6(folder="data/Tests/test6"):
  os.makedirs(folder, exist_ok=True)
  test_med(2, folder)
  with open(f"{folder}/ratings_train.csv", "a") as file:
    rate(file, [1,2,3,4,5], [7,8,9], MAX_RATING)

def test7(folder="data/Tests/test7"):
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, list(range(0,6)), [0], 0)
    rate(file, list(range(0,6)), [1], 1)
    rate(file, list(range(0,6)), [2], 2)
    rate(file, list(range(0,6)), [3], 3)
    rate(file, list(range(0,6)), [4], 4)
    rate(file, list(range(0,6)), [5], 5)
  with open(f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING, 1000)

def test8(folder="data/Tests/test8"):
  os.makedirs(folder, exist_ok=True)
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, [0], [0], 0)
    rate(file, [0], [1], 1)
    rate(file, [0], [2], 2)
    rate(file, [0], [3], 3)
    rate(file, [0], [4], 4)
    rate(file, [0], [5], 5)
    rate(file, list(range(1,6)), [0,6,12], 0)
    rate(file, list(range(1,6)), [1,7,13], 1)
    rate(file, list(range(1,6)), [2,8,14], 2)
    rate(file, list(range(1,6)), [3,9,15], 3)
    rate(file, list(range(1,6)), [4,10,16], 4)
    rate(file, list(range(1,6)), [5,11,17], 5)
  with open(f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING, 1000)

def test9(folder="data/Tests/test9"):
  os.makedirs(folder, exist_ok=True)
  with open(f"{folder}/ratings_train.csv", "w") as file:
    for i in range(0,6):
      rate(file, [i], [i], MAX_RATING)
      if (i-1) >= 0: rate(file, [i], [i-1], MAX_RATING-1)
      if (i+1) <= 5: rate(file, [i], [i+1], MAX_RATING-1)
      if (i-2) >= 0: rate(file, [i], [i-2], MAX_RATING-2)
      if (i+2) <= 5: rate(file, [i], [i+2], MAX_RATING-2)
      if (i-3) >= 0: rate(file, [i], [i-3], MAX_RATING-3)
      if (i+3) <= 5: rate(file, [i], [i+3], MAX_RATING-3)
      if (i-4) >= 0: rate(file, [i], [i-4], MAX_RATING-4)
      if (i+4) <= 5: rate(file, [i], [i+4], MAX_RATING-4)
      if (i-5) >= 0: rate(file, [i], [i-5], MAX_RATING-5)
      if (i+5) <= 5: rate(file, [i], [i+5], MAX_RATING-5)
  with open(f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING, 1000)

def test10(folder="data/Tests/test10"):
  os.makedirs(folder, exist_ok=True)
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, [100], [100], 0)
    rate(file, [100], [101], 1)
    rate(file, [100], [102], 2)
    rate(file, [100], [103], 3)
    rate(file, [100], [104], 4)
    rate(file, [100], [105], 5)
    rate(file, list(range(0,6)), [0,6,12], 0)
    rate(file, list(range(0,6)), [1,7,13], 1)
    rate(file, list(range(0,6)), [2,8,14], 2)
    rate(file, list(range(0,6)), [3,9,15], 3)
    rate(file, list(range(0,6)), [4,10,16], 4)
    rate(file, list(range(0,6)), [5,11,17], 5)
  with open(f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING, 1000)

def generar_tests_csvs():
  test1()
  test2()
  test3()
  test4()
  test5()
  test6()
  test7()
  test8()

print("Generando test data")
test10()
#generar_tests_csvs()