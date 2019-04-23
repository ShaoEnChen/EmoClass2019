import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='plot')
parser.add_argument('--state', type=str, help='path of state')
parser.add_argument('--suffix', type=str, default='', help='suffix of figure name')
args = parser.parse_args()

def plot(state_path, suffix=''):
    state = torch.load(state_path, map_location='cpu')
    acc_history = state['acc_history']
    loss_history = state['loss_history']
    epoch = np.arange(1, len(acc_history) + 1)
    
    fig = plt.figure()
    plt.plot(epoch, acc_history)
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy(%)')
    fig.savefig('validation_accuracy_{}.png'.format(suffix))
    
if __name__ == '__main__':
    plot(args.state, args.suffix)

