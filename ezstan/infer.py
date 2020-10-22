from mynumpy import *
import scipy

def laplace(logp,nsamps):
    z = logp.z0()

    print('logp0',logp(z))
    obj = value_and_grad(lambda z : -logp(z),0)
    #print('obj0',obj(z))
    options = {'disp' : False, 'maxiter' : 10000, 'gtol' : 1e-15} # 'ftol' : 1e-10
    rez = scipy.optimize.minimize(obj,z,method='BFGS',jac=True,options=options)
    #print(rez)
    print('logp1',logp(rez['x']))

    try:
        u = rez.x
        M = cholesky(rez.hess_inv)
        samps = (M @ randn(len(z),nsamps) + expand_dims(u,axis=1)).T
    except:
        samps = np.repeat(np.expand_dims(u,axis=1).T,nsamps,axis=0)
    return samps
    #return logp.constrain(samps)


    # H = hessian(logp.logp)(u)

    # Stan doesn't have support for second order...
    g = autograd.grad(logp)
    eps = 1e-8
    H = zeros((len(z),len(z)))
    for i in range(len(z)):
        z_pos = z+0.0
        z_pos[i] += eps
        z_neg = z+0.0
        z_neg[i] -= eps
        H[:,i] = (g(z_pos)-g(z_neg))/(2*eps)

    #import IPython
    #IPython.embed()

    try:
        M = inv(cholesky(-H)).T
        samps = (M @ randn(len(z),nsamps) + expand_dims(u,axis=1)).T
    except:
        # just return samples at the mean
        samps = np.repeat(np.expand_dims(u,axis=1).T,nsamps,axis=0)

    # keep the samps in z space
    return samps
    #return logp.constrain(samps)

    #return u, M
