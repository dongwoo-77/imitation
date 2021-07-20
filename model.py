from pyrep.objects.vision_sensor import _pixel_to_world_coords
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
import torchvision.models as models

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import csv

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 66 x 200
            nn.Conv2d(3, 24, 5, stride=2, bias=False),
            #nn.ELU(0.2, inplace=True),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(36),
            
            nn.Conv2d(36, 48, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(48),
            
            nn.Conv2d(48, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.Dropout(p=0.4)
        )
        self.linear_layers = nn.Sequential(
            #input from sequential conv layers
            nn.Linear(in_features=64*1*18, out_features=100, bias=False),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50, bias=False),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10, bias=False),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=3, bias=False))
        self._initialize_weights()
        
    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(output.size(0), 64*1*18)
        output = self.linear_layers(output)
        return output


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class mydataset(Dataset): 
    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2
        self.img = list(sorted(os.listdir(os.path.join(path1, "set_reach_b"))))
        self.data = pd.read_csv(path2, header=None)
        print(len(self.img))

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path1, "set_reach_b", self.img[idx])
        img = Image.open(img_path).convert("RGB")
        print(img_path.split('_')[-1].split('.')[0])
        # print(self.data.loc[int(img_path.split('_')[-1].split('.')[0])][0], self.data.loc[int(img_path.split('_')[-1].split('.')[0])][1], self.data.loc[int(img_path.split('_')[-1].split('.')[0])][2])
        # exit()
        label = [self.data.loc[int(img_path.split('_')[-1].split('.')[0])][0], self.data.loc[int(img_path.split('_')[-1].split('.')[0])][1], self.data.loc[int(img_path.split('_')[-1].split('.')[0])][2]]
        label_m = []
        for i in range(len(self.data)):
            label.append([self.data.loc[i][0], self.data.loc[i][1], self.data.loc[i][2]])
            if i > 0:
                label_m.append([self.data.loc[i-1][0] - self.data.loc[i][0], self.data.loc[i-1][1] - self.data.loc[i][1], self.data.loc[i-1][2] - self.data.loc[i][2]])

        return img, label


def main():
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# exit()
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 8

    train_dataset = mydataset('/home/nam/workspace/imitation/data', '/home/nam/workspace/imitation/data/position/set_reach_b.csv')
    test_dataset = mydataset('/home/nam/workspace/imitation/test_data', '/home/nam/workspace/imitation/test_data/position/set_reach_b.csv')

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # train_dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # test_dataset = torch.utils.data.Subset(dataset_test, indices[-50:])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)

    model = mymodel()

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss().to(device)


    train_loss_arr = []
    train_acc_arr = []

    val_loss_arr = []
    val_acc_arr = []


    for epoch in range(num_epochs):

        model.train()

        losses = AverageMeter()
        top1 = AverageMeter()

        for i, (data, target) in enumerate(train_loader):
            
            data = data.to(device)
            target = target.to(device)
            # print(data, target)

            output = model(data) 
            print(target)

            loss = criterion(output, target)

            output.float()
            loss.float()

            prec1 = accuracy(output.data, target)
            prec1 = prec1[0]

            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))

            if i % 100 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        train_loss_arr.append(losses.avg)
        train_acc_arr.append(top1.avg)
        print("train result: Loss: {}, Acc: {}\n".format(losses.avg, top1.avg))


        model.eval()
        with torch.no_grad():
            val_loss_sum = 0
            val_acc_sum = 0

            losses = AverageMeter()
            top1 = AverageMeter()

            for i, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)

                output = model(data) 

                loss = criterion(output, target)

                output.float()
                loss.float()

                prec1 = accuracy(output.data, target)

                prec1 = prec1[0]
                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))

                if i % 100 == 0:
                    print('Test: [{0}/{1}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            i, len(test_loader), loss=losses, top1=top1))

            val_loss_arr.append(losses.avg)
            val_acc_arr.append(top1.avg)
            print("Validation result: Loss: {}, Acc: {}\n".format(losses.avg, top1.avg))

if __name__ == "__main__":
    main()