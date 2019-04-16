import argparse
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import transforms as transforms
from fer2013 import FER2013
from models import *
import utils

parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--model', type=str, default='VGG19', help='network architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='dataset')
parser.add_argument('--bs', type=int, default=64, help='batch size for train')
parser.add_argument('--bs-vt', type=int, default=8, help='batch size for validation / test')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--save-path', type=str, default='checkpoints/best_model.t7', help='path to save model')

# Preprocessing
parser.add_argument('--blur', type=bool, default=False, help='Preprocess: whether to blur inputs')
parser.add_argument('--gs-blur', type=bool, default=False, help='Preprocess: whether to gaussian blur inputs')
parser.add_argument('--sharpen', type=bool, default=False, help='Preprocess: whether to sharpen inputs')
parser.add_argument('--landmark', type=bool, default=False, help='Preprocess: whether to add facial landmarks')
parser.add_argument('--angle-correct', type=bool, default=False, help='Preprocess: whether to do face angle correction')
parser.add_argument('--gamma-correct', type=bool, default=False, help='Preprocess: whether to do gamma correction')
parser.add_argument('--gamma', type=int, default=0.5, help='Preprocess: gamma value for correction')
parser.add_argument('--hist-equal', type=bool, default=False, help='Preprocess: whether to do histogram equalization')
parser.add_argument('--upscale', type=bool, default=False, help='Preprocess: whether to quadruple input pixels')

args = parser.parse_args()

if args.upscale:
    crop_size = 188
    train_file = 'data/train_x4.csv'
    val_file = 'data/validation_x4.csv'
    test_file = 'data/test_x4.csv'
else:
    crop_size = 44
    train_file = 'data/train.csv'
    val_file = 'data/validation.csv'
    test_file = 'data/test.csv'

use_cuda = torch.cuda.is_available()

learning_rate_decay_start = 80
learning_rate_decay_every = 5
learning_rate_decay_rate = 0.9

best_val_acc = 0.0
best_val_acc_epoch = 0

start_epoch = 0
total_epoch = 250

if not os.path.exists(os.path.dirname(args.save_path)):
    os.makedirs(os.path.dirname(args.save_path))

# Prepare data
print('Preparing data...')

def read_data(file_name):
    with open(file_name, 'r') as f:
        data = list(csv.reader(f))
        data = [[d[0], int(d[1])] for d in data]
        random.shuffle(data)
    return data

transform_train = [
    transforms.RandomCrop(crop_size),
    transforms.RandomHorizontalFlip(),
]

transform_test = [
    transforms.TenCrop(crop_size),
]

# Add preprocessing transformations according to the arguments
if args.blur == True:
    transform_train.append(transforms.Blur())
    transform_test.append(transforms.Lambda(lambda crops: [transforms.Blur()(crop) for crop in crops]))
if args.gs_blur == True:
    transform_train.append(transforms.GaussianBlur())
    transform_test.append(transforms.Lambda(lambda crops: [transforms.GaussianBlur()(crop) for crop in crops]))
if args.sharpen == True:
    transform_train.append(transforms.Sharpen())
    transform_test.append(transforms.Lambda(lambda crops: [transforms.Sharpen()(crop) for crop in crops]))
if args.landmark == True:
    transform_train.append(transforms.FacialLandmark())
    transform_test.append(transforms.Lambda(lambda crops: [transforms.FacialLandmark()(crop) for crop in crops]))
if args.angle_correct == True:
    transform_train.append(transforms.RotationByEyesAngle())
    transform_test.append(transforms.Lambda(lambda crops: [transforms.RotationByEyesAngle()(crop) for crop in crops]))
if args.gamma_correct == True:
    transform_train.append(transforms.GammaCorrection(args.gamma))
    transform_test.append(transforms.Lambda(lambda crops: [transforms.GammaCorrection(args.gamma)(crop) for crop in crops]))
if args.hist_equal == True:
    transform_train.append(transforms.HistogramEqualization())
    transform_test.append(transforms.Lambda(lambda crops: [transforms.HistogramEqualization()(crop) for crop in crops]))

transform_train.append(transforms.ToTensor())
transform_test.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))

transform_train = transforms.Compose(transform_train)
transform_test = transforms.Compose(transform_test)

train_set = FER2013(read_data(train_file), transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=1)
val_set = FER2013(read_data(val_file), transform=transform_test)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=1)
test_set = FER2013(read_data(test_file), transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=1)

# Build model
print('Building model...')

if args.model == 'VGG19':
    net = VGG('VGG19', args.upscale)
elif args.model  == 'Resnet18':
    net = ResNet18()

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

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

# Test
def test():
    print('Testing...')
    checkpoint = torch.load(args.save_path)
    if use_cuda:
        net.load_state_dict(checkpoint['net'])
    else:
        net = checkpoint['net']
    net.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
        loss = criterion(outputs_avg, targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs_avg.detach(), 1)
        total += targets.size(0)
        correct += predicted.eq(targets.detach()).cpu().sum().item()
        utils.progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc: {:.3f}% ({:.0f}/{:.0f})'\
                           .format(test_loss / (batch_idx + 1), correct / total * 100, correct, total))

    return correct / total * 100

for epoch in range(start_epoch, total_epoch):
    print('Epoch: {}'.format(epoch))
    train(epoch)
    val(epoch)

print('best_val_acc: {:.3f}%'.format(best_val_acc))
print('best_val_acc_epoch: {}'.format(best_val_acc_epoch))
print('test_acc: {:.3f}%'.format(test()))
