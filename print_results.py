import os
import glob

for subdir in glob.glob(os.path.join("./experiments","*/")):
    exp_name = subdir.split("/")[-2]
    if not os.path.exists(os.path.join(subdir, "report.txt")):
        continue
    with open(os.path.join(subdir, "report.txt"), "r"):
        lines = f.readlines()
    exp_res = lines[2]
    print(exp_name, exp_res)