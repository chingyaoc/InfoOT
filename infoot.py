"""
InfoOT solver
"""
# Author: Ching-Yao Chuang <cychuang@mit.edu>
# License: MIT License

import numpy as np
import scipy.io
import ot
from tqdm import tqdm
from sklearn.metrics import pairwise_distances


def dist(z1, z2, delta=5000):
    x1, x2 = z1[:-1], z2[:-1]
    y1, y2 = z1[-1], z2[-1]
    if y1 != y2:
        return np.linalg.norm(x1 - x2) + delta
    else:
        return np.linalg.norm(x1 - x2)

def ratio(P, Kx, Ky):
    '''
    compute the ratio berween joint and marginal densities
    Parameters
    ----------
    P : transportation plan
    Kx: source kernel matrix
    Ky: target kernel matrix

    Returns
    ----------
    ratio matrix for (x_i, y_j)
    '''
    f_x = Kx.sum(1) / Kx.shape[1]
    f_y = Ky.sum(1) / Ky.shape[1]
    f_x_f_y = np.outer(f_x, f_y)
    constC = np.zeros((len(Kx), len(Ky)))
    f_xy = -ot.gromov.tensor_product(constC, Kx, Ky, P)
    return f_xy / f_x_f_y

def compute_kernel(Cx, Cy, h):
    '''
    compute Gaussian kernel matrices
    Parameters
    ----------
    Cx: source pairwise distance matrix
    Cy: target pairwise distance matrix
    h : bandwidth
    Returns
    ----------
    Kx: source kernel
    Ky: targer kernel
    '''
    std1 = np.sqrt((Cx**2).mean() / 2)
    std2 = np.sqrt((Cy**2).mean() / 2)
    h1 = h * std1
    h2 = h * std2
    # Gaussian kernel (without normalization)
    Kx = np.exp(-(Cx / h1)**2 / 2)
    Ky = np.exp(-(Cy / h2)**2 / 2)
    return Kx, Ky

def migrad(P, Kx, Ky):
    '''
    compute the ratio berween joint and marginal densities
    Parameters
    ----------
    P : transportation plan
    Ks: source kernel matrix
    Kt: target kernel matrix

    Returns
    ----------
    negative gradient w.r.t. MI
    '''
    f_x = Kx.sum(1) / Kx.shape[1]
    f_y = Ky.sum(1) / Ky.shape[1]
    f_x_f_y = np.outer(f_x, f_y)
    constC = np.zeros((len(Kx), len(Ky)))
    # there's a negative sign in ot.gromov.tensor_product
    f_xy = -ot.gromov.tensor_product(constC, Kx, Ky, P)
    P_f_xy = P / f_xy
    P_grad = -ot.gromov.tensor_product(constC, Kx, Ky, P_f_xy)
    P_grad = np.log(f_xy / f_x_f_y) + P_grad
    return -P_grad

def projection(P, X):
    '''
    compute the projection based on similarity matrix
    Parameters
    ----------
    P : transportation plan or similarity matrix
    X : target data

    Returns
    ----------
    projected source data
    '''
    weights = np.sum(P, axis = 1)
    X_proj = np.matmul(P, X) / weights[:, None]
    return X_proj

class FusedInfoOT():
    '''
    Solver for Fused InfoOT
    Parameters
    ----------
    Xs: source data
    Xt: target data 
    h : bandwidth
    Ys: source label
    lam: weight for mutual information
    reg: weight for entropic regularization
    '''
    def __init__(self, Xs, Xt, h, Ys=None, lam=100., reg=1.0):
        self.Xs = Xs
        self.Xt = Xt
        self.Ys = Ys
        self.h = h
        self.lam = lam
        self.reg = reg

        # init kernel
        self.C = pairwise_distances(Xs, Xt)
        if Ys is not None:
            Zs = np.concatenate((Xs, Ys.reshape(-1, 1)), axis=1)
            self.Cs = pairwise_distances(Zs, Zs, metric=dist)
        else:
            self.Cs = pairwise_distances(Xs, Xs)
        self.Ct = pairwise_distances(Xt, Xt)
        self.Ks, self.Kt = compute_kernel(self.Cs, self.Ct, h)
        self.P = None

    def solve(self, numIter=50, verbose='True'):
        '''
        solve projected gradient descent via sinkhorn iteration
        '''
        p = np.zeros(len(self.Xs)) + 1. / len(self.Xs)
        q = np.zeros(len(self.Xt)) + 1. / len(self.Xt)
        P = np.outer(p, q)
        if verbose:
            print('solve projected gradient descent...')
            for i in tqdm(range(numIter)):
                grad_P = migrad(P, self.Ks, self.Kt)
                P = ot.bregman.sinkhorn(p, q, self.C + self.lam * grad_P, reg=self.reg)
        else:
            for i in range(numIter):
                grad_P = migrad(P, self.Ks, self.Kt)
                P = ot.bregman.sinkhorn(p, q, self.C + self.lam * grad_P, reg=self.reg)
        self.P = P
        return P

    def project(self, X, method='barycentric', h=None):
        if method not in ['conditional', 'barycentric']:
            raise Exception('only suppot conditional or barycebtric projection')
        if self.P is None:
            raise Exception('please run FusedInfoOT.solve() to obtain transportation plan')

        if h is None:
            h = self.h

        if np.array_equal(X, self.Xs):
            if method == 'conditional':
                if h == self.h:
                    P = ratio(self.P, self.Ks, self.Kt)
                else:
                    _Ks, _Kt = compute_kernel(self.Cs, self.Ct, h)
                    P = ratio(self.P, _Ks, _Kt)
            else:
                P = self.P
            return projection(P, self.Xt)
        else:
            if method == 'conditional':
                _Cs = pairwise_distances(X, Xs)
                _Ct = pairwise_distances(Xt, Xt)
                _Ks, _Kt = compute_kernel(_Cs, _Ct, h)

                P = ratio(P, _Ks, _Kt)
                return projection(P, self.Xt)
            else:
                raise Exception('barycentric cannot generalize to new samples')

    def conditional_score(self, X, h=None):
        if h is None:
            h = self.h
        _Cs = pairwise_distances(X, self.Xs)
        _Ct = pairwise_distances(self.Xt, self.Xt)
        _Ks, _Kt = compute_kernel(_Cs, _Ct, h)
        return ratio(self.P, _Ks, _Kt)


class InfoOT():
    '''
    Solver for InfoOT. Source and target can have different dimension.
    Parameters
    ----------
    Xs: source data
    Xt: target data
    h : bandwidth
    reg: weight for entropic regularization
    '''
    def __init__(self, Xs, Xt, h, reg=0.05):
        self.Xs = Xs
        self.Xt = Xt
        self.h = h
        self.lam = lam
        self.reg = reg

        # init kernel
        self.Cs = pairwise_distances(Xs, Xs)
        self.Ct = pairwise_distances(Xt, Xt)
        self.Ks, self.Kt = compute_kernel(self.Cs, self.Ct, h)
        self.P = None

    def solve(self, numIter=100, verbose='True'):
        '''
        solve projected gradient descent via sinkhorn iteration
        '''
        p = np.zeros(len(self.Xs)) + 1. / len(self.Xs)
        q = np.zeros(len(self.Xt)) + 1. / len(self.Xt)
        P = np.outer(p, q)
        if verbose:
            print('solve projected gradient descent...')
            for i in tqdm(range(numIter)):
                grad_P = migrad(P, self.Ks, self.Kt)
                P = ot.bregman.sinkhorn(p, q, grad_P, reg=self.reg)
        else:
            for i in range(numIter):
                grad_P = migrad(P, self.Ks, self.Kt)
                P = ot.bregman.sinkhorn(p, q, grad_P, reg=self.reg)
        self.P = P
        return P

    def project(self, X, method='barycentric', h=None):
        if method not in ['conditional', 'barycentric']:
            raise Exception('only suppot conditional or barycebtric projection')
        if self.P is None:
            raise Exception('please run InfoOT.solve() to obtain transportation plan')

        if h is None:
            h = self.h

        if np.array_equal(X, self.Xs):
            if method == 'conditional':
                if h == self.h:
                    P = ratio(self.P, self.Ks, self.Kt)
                else:
                    _Ks, _Kt = compute_kernel(self.Cs, self.Ct, h)
                    P = ratio(self.P, _Ks, _Kt)
            else:
                P = self.P
            return projection(P, self.Xt)
        else:
            if method == 'conditional':
                _Cs = pairwise_distances(X, Xs)
                _Ct = pairwise_distances(Xt, Xt)
                _Ks, _Kt = compute_kernel(_Cs, _Ct, h)

                P = ratio(P, _Ks, _Kt)
                return projection(P, self.Xt)
            else:
                raise Exception('barycentric cannot generalize to new samples')

    def conditional_score(self, X, h=None):
        if h is None:
            h = self.h
        _Cs = pairwise_distances(X, self.Xs)
        _Ct = pairwise_distances(self.Xt, self.Xt)
        _Ks, _Kt = compute_kernel(_Cs, _Ct, h)
        return ratio(self.P, _Ks, _Kt)
