
import autograd.numpy as np
from autograd.numpy import trace, array, pi, e, eye, diag, sqrt, triu, tril, outer, inf, allclose, sin, cos, reshape, repeat, mean, cov, std, zeros, ones, inner, exp, log, log2, meshgrid, concatenate, ceil, floor, expand_dims, isnan, nan, unique, arange
from autograd.numpy.linalg import solve, inv, slogdet, det,  cholesky, eig, eigh, norm #svd
import autograd.scipy.linalg as spla
from numpy.random import rand, randn, seed, multivariate_normal, permutation
import autograd.scipy.stats.multivariate_normal as mvn
from scipy.stats import invwishart
from scipy.integrate import nquad
import autograd
import autograd.util
from time import time, sleep, clock
#from autograd.core import getval
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import grad, value_and_grad, jacobian, hessian

logdet = lambda A : slogdet(A)[1]

def svd(*args,**kwargs):
    return spla.svd(*args, lapack_driver='gesvd', **kwargs)

np.set_printoptions(precision=4,suppress=True,linewidth=100)

#from IPython import embed

from autograd.extend import primitive, defvjp, defvjp_argnum

@primitive
def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x)
    return np.log(np.sum(np.exp(x - max_x))) + max_x
def logsumexp_vjp(ans, x):
    x_shape = x.shape
    return lambda g: np.full(x_shape, g) * np.exp(x - np.full(x_shape, ans))
defvjp(logsumexp, logsumexp_vjp)

@primitive
def logsumexp1(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x,axis=1)
    return np.log(np.sum(np.exp(x - expand_dims(max_x,axis=1)),axis=1)) + max_x
def logsumexp1_vjp(ans, x):
    x_shape = x.shape
    return lambda g: np.full(x_shape, g) * np.exp(x - np.full(x_shape, ans))
defvjp(logsumexp1, logsumexp1_vjp)

def list_logsumexp(A):
    mm = A[0]
    for i in range(1,len(A)):
        mm = np.maximum(mm,A[i])
    rez = exp(A[0]-mm)
    for i in range(1,len(A)):
        rez += exp(A[i]-mm)
    return log(rez)+mm
