import os
import glob
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import pandas as pd

def report_res(subdir, exp_name, is_majority=False, print_option=1):
    f1_list = []
    precision_list = []
    recall_list = []
    acc_list = []
    auc_list = []
    for fold in [0,1,2,3,4]:
        df = pd.read_csv(os.path.join(subdir,f"fold_{fold}_results.csv"))

        if is_majority:
            majority = 0
            majority_count = 0
            for tmp_label in df["gold_labels"].unique():
                if len(df[df["gold_labels"] == tmp_label]) > majority_count:
                    majority = tmp_label
                    majority_count = len(df[df["gold_labels"] == tmp_label])
            if ("persuasion" in exp_name) or ("content" in exp_name):
                df["predicted_labels"] = majority
            else:
                df["predicted_labels"] = 1

        predicted_labels = df["predicted_labels"]
        gold_labels = df["gold_labels"]
        predicted_probs = df["probabilities"]

        fold_metrics = classification_report(gold_labels, predicted_labels, output_dict=True, digits=4)
        if ("persuasion" in exp_name) or ("content" in exp_name):
            fold_f1 = fold_metrics["macro avg"]['f1-score']
            fold_precision = fold_metrics["macro avg"]['precision']
            fold_recall = fold_metrics["macro avg"]['recall']
            fold_acc = fold_metrics["accuracy"]
            fold_auc = 0
        else:
            fold_f1 = fold_metrics["1.0"]['f1-score']
            fold_precision = fold_metrics["1.0"]['precision']
            fold_recall = fold_metrics["1.0"]['recall']
            fold_acc = fold_metrics["accuracy"]
            fold_auc = roc_auc_score(gold_labels, predicted_probs)

        # print(classification_report(gold_labels, predicted_labels, digits=4))

        f1_list.append(fold_f1)
        precision_list.append(fold_precision)
        recall_list.append(fold_recall)
        acc_list.append(fold_acc)
        auc_list.append(fold_auc)

    if print_option == 0:
        m_f1, m_precision, m_recall, m_acc, m_auc = np.round(np.mean(f1_list),3), np.round(np.mean(precision_list),3), np.round(np.mean(recall_list),3), np.round(np.mean(acc_list),3), np.round(np.mean(auc_list),3)
        std_f1, std_precision, std_recall, std_acc, std_auc = np.round(np.std(f1_list),2), np.round(np.std(precision_list),2), np.round(np.std(recall_list),2), np.round(np.std(acc_list),2), np.round(np.std(auc_list),2)
        # res_str = f"{exp_name: <30}{m_precision : <10}{m_recall : <10}{m_f1 : <10}{m_acc : <10}"
        # res_str = f"{m_precision}({std_precision}) {m_recall}({std_recall}) {m_f1}({std_f1}) {m_auc}({std_auc}) {m_acc}({std_acc})"
        res_str = f"${m_precision}_(\pm{std_precision})$ ${m_recall}_(\pm{std_recall})$ ${m_f1}_(\pm{std_f1})$ ${m_auc}_(\pm{std_auc})$ ${m_acc}_(\pm{std_acc})$"

    else:
        m_f1, m_precision, m_recall, m_acc, m_auc = np.round(np.mean(f1_list)*100,2), np.round(np.mean(precision_list)*100,2), np.round(np.mean(recall_list)*100,2), np.round(np.mean(acc_list)*100,2), np.round(np.mean(auc_list)*100,2)
        std_f1, std_precision, std_recall, std_acc, std_auc = np.round(np.std(f1_list)*100,2), np.round(np.std(precision_list)*100,2), np.round(np.std(recall_list)*100,2), np.round(np.std(acc_list)*100,2), np.round(np.std(auc_list)*100,2)

        res_str = f"{m_precision} {m_recall} {m_f1} {m_auc} {m_acc}"

    print(f"{exp_name} 5 fold validation...")
    print(f"{'precision' : <10}{'recall' : <10}{'f1' : <10}{'auc' : <10}{'acc' : <10}")
    print(f"{m_precision}({std_precision}) {m_recall}({std_recall}) {m_f1}({std_f1}) {m_auc}({std_auc}) {m_acc}({std_acc}) ")

    return res_str

def get_res_table():
    exp_res_lsit = []
    exp_res_majority_lsit = []

    for subdir in glob.glob(os.path.join("./experiments","*/")):
        exp_name = subdir.split("/")[-2]
        if not os.path.exists(os.path.join(subdir, "report.txt")):
            continue
        # with open(os.path.join(subdir, "report.txt"), "r") as f:
        #     lines = f.readlines()
        # exp_res = f"{exp_name}{'': <10}{lines[2][:-1]}"
        exp_res = report_res(subdir, exp_name, is_majority=False, print_option=0)
        exp_majority_res = report_res(subdir, exp_name, is_majority=True, print_option=0)

        exp_res = exp_res.split(" ")
        exp_majority_res = exp_majority_res.split(" ")
        res_str = f"{exp_name} & {exp_res[0]} & {exp_res[1]} & {exp_res[2]} & {exp_res[3]}"
        res_majority_str = f"{exp_name} & {exp_majority_res[0]} & {exp_majority_res[1]} & {exp_majority_res[2]} & {exp_majority_res[3]}"
        exp_res_lsit.append(res_str)
        exp_res_majority_lsit.append(res_majority_str)



    for exp_res in sorted(exp_res_lsit):
        print(exp_res)

    print("\n-------majority----------")
    for exp_res in sorted(exp_res_majority_lsit):
        print(exp_res)

def get_data_stas():
    df = pd.read_csv("./data/gun_control_annotation.csv")
    persuasiveness = df["persuasiveness"]
    for threhold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        label_list = []
        for data in persuasiveness:
            if data >= threhold:
                label = 1
            else:
                label = 0
            label_list.append(label)
        ratio = sum(label_list)/len(label_list)
        print(f"threshold {threhold} ratio {ratio}")



get_res_table()
get_data_stas()