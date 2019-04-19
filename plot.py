import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='plot')
parser.add_argument('--state', type=string, help='path of state')
args = parser.parse_args()

def plot(state):
    acc_history = state['acc_history']
    loss_history = state['loss_history']
    epoch = np.arange(1, len(acc_history) + 1)
    
    fig = plt.figure()
    plt.plot(epoch, acc_history)
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy(%)')
    fig.savefig('validation_accuracy.png')
    
    fig = plt.figure()
    plt.plot(epoch, loss_history)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    fig.savefig('loss.png')

if __name__ == '__main__':
    plot(torch.load(args.state))

