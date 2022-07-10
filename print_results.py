import os
import glob

exp_res_lsit = []
for subdir in glob.glob(os.path.join("./experiments","*/")):
    exp_name = subdir.split("/")[-2]
    if not os.path.exists(os.path.join(subdir, "report.txt")):
        continue
    with open(os.path.join(subdir, "report.txt"), "r") as f:
        lines = f.readlines()
    exp_res = f"{exp_name}{'': <10}{lines[2][:-1]}"
    exp_res_lsit.append(exp_res)

for exp_res in sorted(exp_res_lsit):
    print(exp_res)