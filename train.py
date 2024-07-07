import os
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import True_

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from dataset import DMIDataset
from tqdm import tqdm

import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import pandas as pd

from model import DMInet

def get_args():
    parser = argparse.ArgumentParser(description='Train the Network for TIANCHI Challenge',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', metavar='E', type=int, default= 2,
                        help='id of gpu', dest='gpu')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default = 1,# 200,
                        help='Training epochs', dest='epochs')
    parser.add_argument('-bs', '--batch_size', metavar='E', type=int, default = 4, #16,
                        help='Training batch size', dest='batch_size')
    parser.add_argument('-m', '--model_name', metavar='E', type=str, default = "resnet34",
                        help='model name', dest='model_name')
    parser.add_argument('-t', '--train_name', metavar='E', type=str, default = '1',
                        help='1, 2, 3, 4', dest='train_name')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    batch_size = args.batch_size
    epochs = args.epochs
    gpu = args.gpu
    model_name = args.model_name
    train_name = args.train_name
    test_name = train_name
    curr_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    out_dir = os.path.join("experiments", curr_time)
    log_dir = os.path.join(out_dir, "log")
    writer = SummaryWriter(log_dir)

    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # dataset_root = "./SCP"
    dataset_root = "./sample_data/SCP/food-101/images"
    dataset_filelists = []
    train_filelists = []
    test_fileliests = []
    filelists = os.listdir(dataset_root)

    for dir in filelists:
        if not os.path.isdir(dir):
            files = os.listdir(dataset_root + "/" + dir)
            for file in files:
                if not os.path.isdir(file):
                    dataset_filelists.append(file)

    train_filelists, valid_filelists = train_test_split(dataset_filelists, test_size=0.2, random_state=42)

    data_transform = {
    'train': transforms.Compose([transforms.Grayscale(1), 
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, ], [0.229, ])]),
    'test': transforms.Compose([ transforms.Grayscale(1),
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, ], [0.229, ])])}

    if train_name == '1': test_name = '2'
    if train_name == '2': test_name = '1'
    
    train_dataset = DMIDataset(dataset_root=dataset_root, filelists = train_filelists, transforms=data_transform["train"], mode = 'train_' + train_name)
    valid_dataset = DMIDataset(dataset_root=dataset_root, filelists = valid_filelists, transforms=data_transform['test'], mode = 'valid_' + train_name)
    # print(train_dataset.__len__)
    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)
    
    model = DMInet(model_name, 2)
    net = model.model
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs/2], gamma=0.1)

    # epochs = 1
    best_acc = 0.0
    best_loss = 0.0
    best_f1 = 0.0
    save_path = './' + model_name + '_' + train_name + '.pth'
    acc_l = []
    loss_l = []
    train_steps = len(train_loader)
    
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data

            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        scheduler.step()

        # validate
        net.eval()
        acc = 0.0
        y_pred = [] 
        y_true = []
        with torch.no_grad():
            val_bar = tqdm(valid_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                y_pred.extend(predict_y.view(-1).detach().cpu().numpy())        
                y_true.extend(val_labels.view(-1).detach().cpu().numpy()) 
      

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = accuracy_score(y_true, y_pred)
        f1_value = f1_score(y_true, y_pred, average='binary')

        # acc_l.append(val_accurate)
        # loss_l.append(running_loss / train_steps)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
            (epoch + 1, running_loss / train_steps, val_accurate))

        writer.add_scalar("val_accurate", val_accurate, epoch)
        writer.add_scalar("loss", running_loss / train_steps, epoch)
        writer.add_scalar("f1_score", f1_value, epoch)

        if val_accurate > best_acc:
            best_acc = val_accurate
            best_f1 = f1_value
            best_loss = running_loss / train_steps
            torch.save(net.state_dict(), save_path)

    print('best performance: acc: %.3f, loss: %.3f, f1: %.3f' % (best_acc, best_loss, best_f1))
    
    if train_name == '1' or train_name == '2':
        
        test_dataset = DMIDataset(dataset_root=dataset_root, filelists = dataset_filelists, transforms=data_transform['test'], mode = 'test_' + test_name)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=0)
        test_model = DMInet(model_name, 2)
        test_net = test_model.model
        test_net.to(device)
        test_net.load_state_dict(torch.load(save_path))
        test_net.eval()

        acc = 0.0
        y_pred = [] 
        y_true = []
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = test_net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]

                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

                y_pred.extend(predict_y.view(-1).detach().cpu().numpy())        
                y_true.extend(test_labels.view(-1).detach().cpu().numpy()) 

        test_accurate = accuracy_score(y_true, y_pred)
        f1_value = f1_score(y_true, y_pred, average='binary')
        print('test_accuracy: %.3f test_f1: %.3f' % (test_accurate, f1_value))
        writer.add_scalar("test_accurate", test_accurate)
        writer.add_scalar("test_f1", f1_value)
