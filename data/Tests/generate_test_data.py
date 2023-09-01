MAX_RATING = 6
MIN_RATING = 1
rate(file, users, movies, rating):
  for u in users:
    for m in movies:
      file.write(f"{u},{m},{rating},999")

test1(folder="data/Tests/test1"):
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, [0], [1,2,3,4,5], MIN_RATING)
    rate(file, file, [1,2,3,4,5], [0], MAX_RATING)
  with open("f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING)

test2(folder="data/Tests/test2"):
  test1(folder)
  with open(f"{folder}/ratings_train.csv", "w+") as file:
    rate(file, [1], [1,2,3,4,5], MIN_RATING)

test3(folder="data/Tests/test3"):
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, [0], [1,2,3,4,5], MAX_RATING)
    rate(file, file, [1,2,3,4,5], [0], MAX_RATING)
  with open("f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING)

test_med(folder, med_rating):
  with open(f"{folder}/ratings_train.csv", "w") as file:
    rate(file, [0], [1,2,3,4,5], med_rating)
    rate(file, [1], [1,2,3,4,5], med_rating)
    rate(file, file, [1,2,3,4,5], [0], MAX_RATING)
  with open("f"{folder}/ratings_test.csv", "w") as file:
    rate(file, [0], [0], MAX_RATING)

test4(folder="data/Tests/test4"):
  test_med(3)

test4(folder="data/Tests/test4"):
  test_med(2)

generar_tests_csvs():
  test1()
  test2()
  test3()
  test4()
  test5()

generar_tests_csvs()
