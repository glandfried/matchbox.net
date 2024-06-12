print("Generating 5 star and binary csvs...")
with open("ml-100k/u.data", "r") as f:
    with open("ml-100k/ratings.csv", "w") as fw:
        with open("ml-100k-binary/ratings.csv", "w") as fw_bin:
            fw.write("userId,movieId,rating,timestamp")
            fw_bin.write("userId,movieId,rating,timestamp")
            for line in f.readlines():
                l = [x for x in line.split("\t")]
                lint = [int(x) for x in line.split("\t")]
                lint[2] = 1 if lint[2] >= 4 else 0
                lint = [str(x) for x in lint]
                fw.write(",".join(l))
                fw_bin.write(",".join(lint))