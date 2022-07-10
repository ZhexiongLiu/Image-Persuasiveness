import torch
import torchvision
import json
import ast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import random
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torchvision import transforms
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
from dataloader import *


def train_model(model, train_dataloaders, val_dataloaders, criterion, optimizer, num_epochs=3):
    best_acc = 0.0
    best_loss = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        ##################### train ##########################
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (_, inputs, labels) in enumerate(train_dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = torch.softmax(logits,dim=-1)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))

        ##################### validation ##########################
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        predicted_labels = []
        predicted_img_ids = []
        gold_labels = []

        for i, (img_ids, inputs, labels) in enumerate(val_dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            outputs = torch.softmax(logits, dim=-1)
            loss = criterion(logits, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            predicted_img_ids += list(img_ids)
            predicted_labels += preds.detach().cpu().tolist()
            gold_labels += labels.detach().cpu().tolist()

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset)
        print('val loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if best_acc < epoch_acc:
            best_acc = epoch_acc
            best_loss = epoch_loss

            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts , 'best_model_weight.pth')

        print('best loss: {:.4f}, acc: {:.4f}'.format(best_loss, best_acc))

        # predicted_img_ids = np.concatenate(predicted_img_ids, axis=1)
        # predicted_labels = np.concatenate(predicted_labels, axis=1)
        # gold_labels = np.concatenate(gold_labels, axis=1)

        print(classification_report(gold_labels, predicted_labels))



    return model


def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()



# define some re-usable stuff
# IMAGE_SIZE = 224
# NUM_CLASSES = 1103
# BATCH_SIZE = 32
device = torch.device("cuda:0")
# make some augmentations on training data
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomResizedCrop((224, 224)),
    # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])



df = pd.read_csv('./data/gun_control_annotation.csv', index_col=0)
kfold = KFold(n_splits=5, shuffle=True, random_state=22)
for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
    print('fold {}...'.format(fold + 1))

    train_annotation = df.iloc[train_idx].reset_index()
    val_annotation = df.iloc[val_idx].reset_index()

    train_dataset = ImageDataset(annotation=train_annotation, root_dir='./data/images', transform=train_transform)
    val_dataset = ImageDataset(annotation=val_annotation, root_dir='./data/images', transform=val_transform)

    train_dataloaders = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=16)
    val_dataloaders = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=16)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ImageModelResNet50()
    # model = ImageModelResNet101()
    # model = ImageModelVGG16()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

    model.apply(reset_weights)
    
    train_model(model, train_dataloaders, val_dataloaders, criterion, optimizer, 20)