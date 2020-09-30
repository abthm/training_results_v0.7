from matplotlib import pyplot as plt
import argparse
import numpy as np

def read_plot(loss_path,acc_path):
    print(type(loss_path))
    print(acc_path)
    loss=np.load(loss_path)
    acc=np.load(acc_path, allow_pickle=True)
    plt.figure(figsize=(10,4))
    plt.plot(loss,label='loss')
    #fig.plot(acc,'g',label=='acc')
    plt.plot(acc,label='mlm_acc')
    plt.xticks(np.arange(0, len(loss), 1000)) 
    plt.yticks(np.arange(0, np.amax(loss), 1))
    plt.legend('loss','acc')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.savefig('mi100-f16-mhalib.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_path')
    parser.add_argument('--acc_path')
    args = parser.parse_args()
    read_plot(args.loss_path,args.acc_path)

