import json
import ast
import torch.nn.init
import glob
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
import torch.utils.data as Data
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init
from sklearn import metrics
from sklearn.model_selection import KFold
from tensorboard_logger import configure, log_value
import argparse
import copy
import time
from utils import *
from models import *


def infer_persuasiveness(id, top_k=5):
    with open(f"./data/text/{id}.txt", "r") as f:
        text = f.read()
        print("text", text)

    if args.data_mode == 1 or args.data_mode == 4:
        text_emb = np.load(f"./data/text_feats_with_captions/{id}.npy")
    else:
        text_emb = np.load(f"./data/text_feats_bert_1024/{id}.npy")

    paths = glob.glob("./data/image_feats_vgg16_1024/*npy")
    img_emb_list = []
    img_id_list = []
    for path in paths:
        img_id = path.split("/")[-1].split(".")[0]
        img_id_list.append(img_id)
        img_emb = np.load(f"./data/image_feats_vgg16_1024/{img_id}.npy")
        img_emb = np.squeeze(img_emb)
        img_emb_list.append(img_emb)

    output_prob_list = []
    for img_emb in img_emb_list:
        img_text = np.concatenate((img_emb, text_emb), axis=0)
        img_text = torch.FloatTensor(img_text).to(device)
        outputs = model_ft(img_text)
        probs = torch.sigmoid(outputs).detach().cpu().tolist()[0]
        output_prob_list.append(probs)

    candidates = sorted([_ for _ in zip(output_prob_list,img_id_list)], reverse=True)
    rank_ids = [id for (prob, id) in candidates]
    rank_prob = [prob for (prob, id) in candidates]

    cand_ids = rank_ids[:top_k]
    cand_probs = rank_prob[:top_k]
    show_image(cand_ids, cand_probs, id, option="persuasiveness")


def infer_image_content():
    pass

def infer_persuasive_mode(id, mode, top_k=5):
    with open(f"./data/text/{id}.txt", "r") as f:
        text = f.read()
        print("text", text)

    if args.data_mode == 1 or args.data_mode == 4:
        text_emb = np.load(f"./data/text_feats_with_captions/{id}.npy")
    else:
        text_emb = np.load(f"./data/text_feats_bert_1024/{id}.npy")

    paths = glob.glob("./data/image_feats_vgg16_1024/*npy")
    img_emb_list = []
    img_id_list = []
    for path in paths:
        img_id = path.split("/")[-1].split(".")[0]
        img_id_list.append(img_id)
        img_emb = np.load(f"./data/image_feats_vgg16_1024/{img_id}.npy")
        img_emb = np.squeeze(img_emb)
        img_emb_list.append(img_emb)

    output_prob_list = []
    for img_emb in img_emb_list:
        img_text = np.concatenate((img_emb, text_emb), axis=0)
        img_text = torch.FloatTensor(img_text).to(device)
        outputs = model_ft(img_text)
        probs = torch.sigmoid(outputs).detach().cpu().tolist()[0]
        output_prob_list.append(probs)

    candidates = sorted([_ for _ in zip(output_prob_list,img_id_list)], reverse=True)
    rank_ids = [id for (prob, id) in candidates]
    rank_prob = [prob for (prob, id) in candidates]

    cand_ids = rank_ids[:top_k]
    cand_probs = rank_prob[:top_k]
    show_image(cand_ids, cand_probs, id, option=f"persuasive mode {mode}")


def get_argparser():
    parser = argparse.ArgumentParser(description='Persuasiveness')
    parser.add_argument('--data-path', default='./data', help='path to data')
    parser.add_argument('--exp-dir', default='./experiments/baselines/', help='path save experimental results')
    parser.add_argument('--exp-mode', default=0, choices=[0,1,2,3,4], type=int, help='0:persuasive; 1:image content; 2:persuasive mode logos; 3:persuasive mode panthos; 4:persuasive mode ethos')
    parser.add_argument('--data-mode', default=3, choices=[3,4], type=int, help='0:text; 1:text+caption; 2:image; 3:image+text; 4:image+text+caption')
    parser.add_argument('--gpus', default='0', type=str, help='specified gpus')
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    device, gpu_ids = get_device(args)
    configure(args.exp_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 0:persuasive; 1:image concent; 2:persuasive mode
    if args.exp_mode == 0: # persuasiveness
        output_size = 1
    elif args.exp_mode == 1: # image content
        output_size = 6
    else: # persuasive mode
        output_size = 1

    # mode 0:text; 1:text+caption; 2:image; 3:image+text; 4:image+text+caption
    if args.data_mode == 0 or args.data_mode == 1:
        input_size = 1024
    elif args.data_mode == 3 or args.data_mode == 4:
        input_size = 2048
        # input_size = 26112
        # input_size = 101376
    else:
        input_size = 1024
        # input_size = 25088
        # input_size = 100352

    model_path = args.exp_dir + f"exp{args.exp_mode}_data{args.data_mode}"

    model_ft = Net(input_size, 512, output_size).to(device)
    checkpoint = torch.load(os.path.join(model_path,"models", "model_fold_4_best_model.pth.tar"))
    print(model_path)
    model_ft.load_state_dict(checkpoint['state_dict'])
    model_ft.eval()


    if args.exp_mode == 0:
        infer_persuasiveness(id=1338553547928047616)
    elif args.exp_mode == 1:
        infer_image_content()
    else:
        infer_persuasive_mode(id=1338553547928047616, mode="ethos")