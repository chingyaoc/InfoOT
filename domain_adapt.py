import numpy as np
import scipy
import argparse

from sklearn.neighbors import KNeighborsClassifier
from infoot import FusedInfoOT

def get_acc(X_train, Y_train, X_test, Y_test):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, Y_train)
    pred = knn.predict(X_test)
    acc = (pred == Y_test).mean()
    print('[!] 1-NN Accuracy: {}'.format(acc))


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

# random shuffle target data
idx = np.array(range(len(X2)))
np.random.shuffle(idx)
X2, Y2 = X2[idx], Y2[idx]

X2_train = X2[:int(len(X2)*0.9)]
Y2_train = Y2[:int(len(X2)*0.9)]
X2_test = X2[int(len(X2)*0.9):]
Y2_test = Y2[int(len(X2)*0.9):]

ot = FusedInfoOT(X1, X2_train, h=0.5, Ys=Y1)
ot.solve()

print('InfoOT Barycentric Proj')
get_acc(ot.project(X1, method='barycentric'), Y1, X2_test, Y2_test)
print('InfoOT Conditional Proj')
get_acc(ot.project(X1, method='conditional'), Y1, X2_test, Y2_test)


