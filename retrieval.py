import numpy as np
import scipy
import argparse

from infoot import FusedInfoOT

def get_acc(score, Y1_test, Y2):
    idx = np.argsort(-score, axis = 1)
    idx_1 = idx[:, :1]
    idx_5 = idx[:, :5]
    idx_15 = idx[:, :15]
    acc_1, acc_5, acc_15 = 0., 0., 0.
    for i in range(len(X1_test)):
        pred_1 = Y2[idx_1[i]]
        pred_5 = Y2[idx_5[i]] 
        pred_15 = Y2[idx_15[i]]
        acc_1 += float(pred_1 == Y1_test[i])
        acc_5 += (pred_5 == Y1_test[i]).mean()
        acc_15 += (pred_15 == Y1_test[i]).mean()
    acc_1 = acc_1 / len(X1_test)
    acc_5 = acc_5 / len(X1_test)
    acc_15 = acc_15 / len(X1_test)
    print('P@1: {}'.format(acc_1))
    print('P@5: {}'.format(acc_5))
    print('P@15: {}'.format(acc_15))
    return acc_1, acc_5, acc_15


parser = argparse.ArgumentParser(description='Domain Adaptation')
parser.add_argument('--src', default='caltech', type=str, help='source dataset name')
parser.add_argument('--tgt', default='dslr', type=str, help='target dataset name')
parser.add_argument('--rs', default=0, type=int, help='random seed')

args = parser.parse_args()
np.random.seed(args.rs)

# load data
mat1 = scipy.io.loadmat('decaf6/'+args.src+'_decaf.mat')
mat2 = scipy.io.loadmat('decaf6/'+args.tgt+'_decaf.mat')

X1 = mat1['feas']
Y1 = mat1['labels'].reshape(-1)
X2 = mat2['feas']
Y2 = mat2['labels'].reshape(-1)

idx = np.array(range(len(X1)))
np.random.shuffle(idx)
X1, Y1 = X1[idx], Y1[idx]

X1_train = X1[:int(len(X1)*0.9)]
Y1_train = Y1[:int(len(X1)*0.9)]
X1_test = X1[int(len(X1)*0.9):]
Y1_test = Y1[int(len(X1)*0.9):]

ot = FusedInfoOT(X1_train, X2, h=0.5, reg=5.)
ot.solve()

acc_1, acc_5, acc_15 = get_acc(ot.conditional_score(X1_test), Y1_test, Y2)
