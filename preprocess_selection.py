import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import transforms as transforms
from fer2013 import FER2013
from models import *
import utils


def shrink_data(train_set,size):

    #{'image': 4000, 'label': 'xyz'}
    image = vars(train_set).get(image)
    label = vars(train_set).get(labels)
    if size > len(image) or size <= 0:
        raiseTypeError("Shrink_data : Wrong size!!")
    return image[0:size], label[0:size]


def train_validation_for_preprocess_selection(train_set,transform_train):
        train_set = FER2013(shrink_data(train_file), transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=1)
        print('Building model...')

        if args.model == 'VGG19':
            net = VGG('VGG19')
        elif args.model  == 'Resnet18':
            net = ResNet18()

        if use_cuda:
            net.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        transform_test = transforms.Compose([
            transforms.TenCrop(crop_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])


        val_set = FER2013(read_data(val_file), transform=transform_test)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=1)

        # Train
        def train(epoch):
            print('Training...')
            net.train()
            train_loss = 0.0
            correct = 0.0
            total = 0.0

            if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
                frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
                decay_factor = learning_rate_decay_rate ** frac
                current_lr = args.lr * decay_factor
                utils.set_lr(optimizer, current_lr)
            else:
                current_lr = args.lr
            print('learning_rate: {}'.format(current_lr))

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                utils.clip_gradient(optimizer, 0.1)
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.detach(), 1)
                total += targets.size(0)
                correct += predicted.eq(targets.detach()).cpu().sum().item()
                utils.progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc: {:.3f}% ({:.0f}/{:.0f})'\
                                   .format(train_loss / (batch_idx + 1), correct / total * 100, correct, total))

        # Do validation
        def val(epoch):
            print('Doing validation...')
            global best_val_acc
            global best_val_acc_epoch
            net.eval()
            val_loss = 0.0
            correct = 0.0
            total = 0.0

            for batch_idx, (inputs, targets) in enumerate(val_loader):
                bs, ncrops, c, h, w = np.shape(inputs)
                inputs = inputs.view(-1, c, h, w)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
                loss = criterion(outputs_avg, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs_avg.detach(), 1)
                total += targets.size(0)
                correct += predicted.eq(targets.detach()).cpu().sum().item()
                utils.progress_bar(batch_idx, len(val_loader), 'Loss: {:.3f} | Acc: {:.3f}% ({:.0f}/{:.0f})'\
                                   .format(val_loss / (batch_idx + 1), correct / total * 100, correct, total))

            # Save checkpoint
            val_acc = correct / total * 100
            if val_acc > best_val_acc:
                print('Saving checkpoint...')
                print('best_val_acc: {:.3f}'.format(val_acc))
                state = {
                    'net': net.state_dict() if use_cuda else net,
                    'acc': val_acc,
                    'epoch': epoch,
                }
                torch.save(state, args.save_path)
                best_val_acc = val_acc
                best_val_acc_epoch = epoch

        for epoch in range(start_epoch, total_epoch):
            print('Epoch: {}'.format(epoch))
            train(epoch)
            val(epoch)

        print('best_val_acc: {:.3f}%'.format(best_val_acc))
        print('best_val_acc_epoch: {}'.format(best_val_acc_epoch))

        return best_val_acc

def select_best_preprocess(object):

    if not isinstance(object ,dict):
        raiseTypeError("Type Error")

    return max(object.items(), key=operator.itemgetter(1))[0]


def grid_search(split_trainset,param_grid ):
    #param_grid = {"HistogramEqualization":[True, False], "RotationByEyesAngle":[True, False], "Blur":["B","G","No Blur"],
    #        "Sharpen":[True, False]}

    combination = [ (h,r,b,s) for h in [True, False] for r in [True, False] for b in ["B","G","No Blur"] for s in [True, False]]
    transform_list = [transforms.RandomCrop(crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()]

    score_table={}
    for combs in combination:
        for item in range(len(combs)):
            if item ==0 and combs[item]==True:
                transform_list.append(transforms.HistogramEqualization())
            if item ==1 and combs[item]==True:
                transform_list.append(transforms.RotationByEyesAngle())
            if item ==2 and combs[item]=="B":
                 transform_list.append(transforms.Blur())
            if item ==2 and combs[item]=="G":
                 transform_list.append(transforms.GaussianBlur())
            if item ==3 and combs[item]==True:
                transform_list.append(transforms.Sharpen())
            transform_train = transforms.Compose(transform_list)

            score_table.update({transform_train,train_validation_for_preprocess_selection(split_trainset,transform_train)})

    print("Grid search finished.")
    print("Score Table: " ,score_table)
    return select_best_preprocess(score_table)
