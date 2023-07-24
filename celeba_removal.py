# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import print_function
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import os
from sklearn.linear_model import LogisticRegression
from utils3 import load_features
#from infl.calc_influence_function import calc_img_wise
#from infl.utils import init_logging, get_default_config
from PIL import Image
import random
from data.attr_dataset import AttributeDataset, AdultDataset
import torchvision
import wandb
import shutil
from data.mlp2 import *
from data.resnet import *
import copy
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser(description='Training a removal-enabled linear model and testing removal')
parser.add_argument('--result-dir', type=str, default='result', help='directory for saving results')
parser.add_argument('--save-dir', type=str, default='result', help='directory for saving results')
parser.add_argument('--extractor', type=str, default='resnet50', help='extractor type')
parser.add_argument('--dataset', type=str, default='SVHN', help='dataset')
parser.add_argument('--lam', type=float, default=1e-6, help='L2 regularization')
parser.add_argument('--std', type=float, default=10.0, help='standard deviation for objective perturbation')
parser.add_argument('--num-removes', type=int, default=1000, help='number of data points to remove')
parser.add_argument('--train-splits', type=int, default=1, help='number of training data splits')
parser.add_argument('--subsample-ratio', type=float, default=1.0, help='negative example subsample ratio')
parser.add_argument('--num-steps', type=int, default=100, help='number of optimization steps')
parser.add_argument('--train-mode', type=str, default='ovr', help='train mode [ovr/binary]')
parser.add_argument('--train-sep', action='store_true', default=False, help='train binary classifiers separately')
parser.add_argument('--verbose', action='store_true', default=False, help='verbosity in optimizer')
parser.add_argument('--resume',action='store_true')
parser.add_argument('--learning_rate', type=float, default=0.00001)
args = parser.parse_args()

device = torch.device("cuda")
from tqdm import tqdm
import pandas as pd

from data.util import get_dataset, IdxDataset, ZippedDataset
data_dir = "/*/celebA"
dataset_tag = 'CelebA'

train_dataset = get_dataset(
    dataset_tag,
    data_dir=data_dir,
    dataset_split="train",
    transform_split="train",
)
valid_dataset = get_dataset(
    dataset_tag,
    data_dir=data_dir,
    dataset_split="eval",
    transform_split="eval",
)



def lr_loss(w, X, y, lam):
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2

def lr_eval(w, X, y):
    return X.mv(w).sign().eq(y).float().mean()

def lr_grad(w, X, y, lam):
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z-1) * y) + lam * X.size(0) * w

def lr_hessian_inv(w, X, y, lam, batch_size=50000):
    z = torch.sigmoid(X.mv(w).mul_(y))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i+1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()

def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-10, verbose=False):
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)
    optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-20)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data

def ovr_lr_loss(w, X, y, lam, weight=None):
    z = batch_multiply(X, w).mul_(y)
    if weight is None:
        return -F.logsigmoid(z).mean(0).sum() + lam * w.pow(2).sum() / 2
    else:
        return -F.logsigmoid(z).mul_(weight).sum() + lam * w.pow(2).sum() / 2

def ovr_lr_optimize(X, y, lam, weight=None, b=None, num_steps=100, tol=1e-10, verbose=False):
    w = torch.autograd.Variable(torch.zeros(X.size(1), y.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return ovr_lr_loss(w, X, y, lam, weight)
        else:
            return ovr_lr_loss(w, X, y, lam, weight) + (b * w).sum() / X.size(0)
    optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-10)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = ovr_lr_loss(w, X, y, lam, weight)
        if b is not None:
            if weight is None:
                loss += (b * w).sum() / X.size(0)
            else:
                loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
        loss.backward()
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data

def batch_multiply(A, B, batch_size=500000):
    if A.is_cuda:
        if len(B.size()) == 1:
            return A.mv(B)
        else:
            return A.mm(B)
    else:
        out = []
        num_batch = int(math.ceil(A.size(0) / float(batch_size)))
        with torch.no_grad():
            for i in range(num_batch):
                lower = i * batch_size
                upper = min((i+1) * batch_size, A.size(0))
                A_sub = A[lower:upper]
                A_sub = A_sub.to(device)
                if len(B.size()) == 1:
                    out.append(A_sub.mv(B).cpu())
                else:
                    out.append(A_sub.mm(B).cpu())
        return torch.cat(out, dim=0).to(device)

def spectral_norm(A, num_iters=20):
    x = torch.randn(A.size(0)).float().to(device)
    norm = 1
    for i in range(num_iters):
        x = A.mv(x)
        norm = x.norm()
        x /= norm
    return math.sqrt(norm)

def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.result_dir, args.save_dir, 'model_best.pth.tar'))


def train(args, epoch, model, criterion, train_loader, optimizer, logging=True):
    model.train()
    nTrain = len(train_loader.dataset) # number of images
    loss_logger = AverageMeter()

    res = list()
    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for batch_idx, (_, images, attr) in enumerate(t):
        #if batch_idx == 100: break # for debugging

        # Set mini-batch dataset
        images = images.cuda()
        targets = attr[:, 9]
        bias_label = attr[:, 20]
        targets = targets.cuda()

        # Forward, Backward and Optimizer
        preds = model(images)

        loss = criterion(preds, targets)
        loss_logger.update(loss.item())

        preds = torch.sigmoid(preds)
        
        res.append((batch_idx, preds.data.cpu(), targets.data.cpu(), targets.data.cpu()))
        # backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        t.set_postfix(loss = loss_logger.avg)

    total_preds   = torch.cat([entry[1] for entry in res], 0)
    total_targets = torch.cat([entry[2] for entry in res], 0)
    total_genders = torch.cat([entry[3] for entry in res], 0)
    
    #print(total_preds)
    total_preds2 = total_preds.max(1)[0]
    
    correct = (total_preds2 == total_targets).long()
    total_correct = correct.sum()
    total_num = correct.shape[0]
    accs = total_correct/float(total_num)
    
    #meanAP = average_precision_score(total_targets.numpy(), total_preds2.numpy(), average='macro')



class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def test(args, epoch, model, val_loader, logging=True):

    model.eval()
    #nVal = len(val_loader.dataset) # number of images
    loss_logger = AverageMeter()

    res = list()
    t = tqdm(val_loader, desc = 'Val %d' % epoch)

    for batch_idx, (_, images, attr) in enumerate(t):
        #if batch_idx == 100: break # for debugging

        # Set mini-batch dataset
        images = images.cuda()
        targets = attr[:, 9]
        bias_label = attr[:, 20].cuda()
        targets = targets.cuda()
        
        
        targets2 = targets[targets == bias_label]
    
        preds = model(images)
        preds2 = preds[targets == bias_label]

        preds = torch.softmax(preds, 1)
        res.append((preds2.data.cpu(), preds.data.cpu(), targets.data.cpu(), targets2.data.cpu()))
        
        



    total_preds   = torch.cat([entry[1] for entry in res], 0)
    total_targets = torch.cat([entry[2] for entry in res], 0)
    
    total_preds2   = torch.cat([entry[0] for entry in res], 0)
    total_targets2 = torch.cat([entry[3] for entry in res], 0)

    total_preds3 = total_preds.argmax(1)
    total_preds4 = total_preds2.argmax(1)
    
    
    correct = (total_preds3 == total_targets).long()
    
    total_correct = correct.sum()
    total_num = correct.shape[0]
    accs = total_correct/float(total_num)
    
    correct = (total_preds4 == total_targets2).long()
    
    total_correct = correct.sum()
    total_num = correct.shape[0]
    accs2 = total_correct/float(total_num)
    
    #print(total_targets.size())
    #print(total_preds.size())

    if not args.resume:
        wandb.log({"accs": accs})

    return accs, accs2

if not args.resume:
    wandb.init(project='modeldebias',entity="*",name='*')

model = resnet20(num_classes=2)
model = model.cuda()

if args.resume:
    if os.path.isfile(os.path.join(args.result_dir, args.save_dir, 'checkpoint.pth.tar')):
        print("=> loading checkpoint '{}'".format(args.save_dir))
        checkpoint = torch.load(os.path.join(args.result_dir, args.save_dir, 'checkpoint.pth.tar'))
        args.start_epoch = 0
        best_performance = checkpoint['best_performance']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.save_dir))
        
if args.resume:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        

def trainable_params():
    for param in model.parameters():
        if param.requires_grad:
            yield param

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('num_trainable_params:', num_trainable_params)
optimizer = torch.optim.Adam(trainable_params(), args.learning_rate, weight_decay = 1e-5)



from torchvision import transforms as T
transforms = {
    "ColoredMNIST": {
        "train": T.Compose([T.ToTensor()]),
        "eval": T.Compose([T.ToTensor()])
        },
}




train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,
        shuffle = True, num_workers = 6, pin_memory = True)

val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 32,
        shuffle = False, num_workers = 4, pin_memory = True)



if not args.resume:

    criterion = nn.CrossEntropyLoss().cuda()

    cur_score, wc_score = test(args, 0, model, val_loader, logging=False)

    best_performance = 0
    for epoch in range(60):
        train(args, epoch, model, criterion, train_loader, optimizer, logging=True)
        cur_score, wc_score = test(args, epoch, model, val_loader, logging=True)
        print("epoch:", epoch, ", acc:", cur_score, ", wc_score:", wc_score)
        is_best = cur_score > best_performance
        best_performance = max(cur_score, best_performance)
        wandb.log({"epoch": epoch})
        wandb.log({"best_performance": best_performance})
        model_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance}
        pathx = os.path.join(args.result_dir, args.save_dir)
        if not os.path.exists(pathx):
            os.mkdir(pathx)
        save_checkpoint(args, model_state, is_best, os.path.join(args.result_dir, args.save_dir, \
                'checkpoint.pth.tar'))

        # save the model from the last epoch
        if epoch == 60:
            torch.save(model_state, os.path.join(args.save_dir, \
                'checkpoint_%d.pth.tar' % args.num_epochs))

if not args.resume:
    exit()
    



best_performance, wc_score = test(args, 0, model, val_loader, logging=True)
print("iter:", 0, ", acc:", best_performance, ", wc_score:", wc_score)


def train_features(train_loader, model):
    res = []
    t = tqdm(train_loader)
    for batch_idx, (_, images, targets) in enumerate(t):
        #if batch_idx == 100: break # for debugging

        # Set mini-batch dataset
        images = images.cuda()
        #print(images.size())
        
        feas = model.extract(images)
        

        res.append(feas)
        
    features = torch.cat([fea for fea in res], 0)
    return features

X_features = train_features(train_loader, model)

triggering_ids1 = []
triggering_ids2 = []

t = tqdm(val_loader)

for batch_idx, (index, images, attr) in enumerate(t):
    #if batch_idx == 100: break # for debugging

    # Set mini-batch dataset
    images = images.cuda()
    targets = attr[:, 9]
    bias_label = attr[:, 20]
    idx1 = index[attr[:,9] == 1]
    idx2 = index[attr[:,20] == 1]
    idx3 = index[attr[:,20] == 0]
    
    idx4 = [int(k.item()) for k in idx1 if k in idx2]
    idx5 = [int(k.item()) for k in idx1 if k in idx3]


    triggering_ids1.extend(idx4)
    triggering_ids2.extend(idx5)
    
print(len(triggering_ids1))
print(len(triggering_ids2))


bias = []
for i in range(100):
    k = 1
    ids1 = triggering_ids1[i]
    ids2 = triggering_ids2[i]

    X_c1 = val_loader.dataset[ids1][1].unsqueeze(0).cuda()
    X_c2 = val_loader.dataset[ids2][1].unsqueeze(0).cuda()
    
    predc1 = model(X_c1)
    predc2 = model(X_c2)
    predc1 = torch.softmax(predc1, 1)
    predc2 = torch.softmax(predc2, 1)
    predc1 = predc1[0][k]
    predc2 = predc2[0][k]
    bias.append((predc1 - predc2).item())

    
bias1 = sum(bias)/len(bias)



import time 
torch.cuda.synchronize()
start = time.time()
for i in range(100):
    ids1 = triggering_ids1[i]
    ids2 = triggering_ids2[i]
    
    X_c1 = val_loader.dataset[ids1][1].unsqueeze(0).cuda()
    X_c2 = val_loader.dataset[ids2][1].unsqueeze(0).cuda()
    
    feature1 = model.extract(X_c1)
    feature2 = model.extract(X_c2)
    
    weights = copy.deepcopy(model.state_dict())
    unfrozed_weights = weights['fc.weight'].permute(1,0)

    
    y_train = train_loader.dataset.attr[:,9].cuda()

    y_c_onehot = torch.tensor(val_loader.dataset[ids1][1]).cuda()
    k = val_loader.dataset[ids1][2]
    k = 1
    H_inv = lr_hessian_inv(unfrozed_weights[:, k], X_features, y_train, args.lam)
    grad_i = lr_grad(unfrozed_weights[:, k], feature1, torch.tensor(1.).unsqueeze(0).cuda(), args.lam)
    grad_i2 = lr_grad(unfrozed_weights[:, k], feature2, torch.tensor(1.).unsqueeze(0).cuda(), args.lam)

    Delta = H_inv.mv(grad_i2 - grad_i)

    unfrozed_weights[:, k] += Delta
    
    weights['fc.weight'] = unfrozed_weights.permute(1,0)
    
    modelx = resnet(num_classes=2)
    modelx = modelx.cuda()
    for param in modelx.parameters():
        param.requires_grad = False
    for param in modelx.fc.parameters():
        param.requires_grad = True

    modelx.load_state_dict(weights)
    
    cur_score, wc_score = test(args, i, modelx, val_loader, logging=True)
    print("iter:", i, ", acc:", best_performance, ", acc2:", cur_score, ", wc_score:", wc_score)
    if cur_score >= best_performance:
        best_performance = cur_score
    
    model.load_state_dict(weights)
    
end = time.time()
training_time = start - end
print("training time: ", training_time)

bias = []
for i in range(100):
    k = 1
    ids1 = triggering_ids1[i]
    ids2 = triggering_ids2[i]

    X_c1 = val_loader.dataset[ids1][1].unsqueeze(0).cuda()
    X_c2 = val_loader.dataset[ids2][1].unsqueeze(0).cuda()
    predc1 = model(X_c1)
    predc2 = model(X_c2)
    predc1 = torch.softmax(predc1, 1)
    predc2 = torch.softmax(predc2, 1)
    predc1 = predc1[0][k]
    predc2 = predc2[0][k]
    bias.append((predc1 - predc2).item())
    
bias2 = sum(bias)/len(bias)

print("bias1: ", "%.4f"%bias1)
print("bias2: ", "%.4f"%bias2)