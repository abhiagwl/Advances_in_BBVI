import pystan
import warnings
from hashlib import md5
import pickle
import re
import autograd
import numpy as np
import os
import collections
import scipy
import warnings

###########################################################################
#  stuff related to loading stan models and data
###########################################################################

file = open('./good_model_paths.pkl','rb')
good_model_paths = pickle.load(file)
file.close()

def eznum():
    return len(good_model_paths)

def ezload(i):
    if i < 0 or i >= eznum():
        print('input i must be 0 <= i < ' + str(eznum()))
        raise ValueError('input i must be 0 <= i < ' + str(eznum()))
    return get_logp(good_model_paths[i])

def all_model_paths():
    data_suffix = '.data.R'
    code_suffix = '.stan'
    
    model_paths = []
    for root, dir, files in os.walk("."):
        for file in files:
            if file.endswith(data_suffix):
                model_name = file[:-len(data_suffix)]
                if os.path.isfile(root+'/'+model_name+code_suffix) and model_name[0:2] != '._': # stupid hidden files...
                    model_paths.append((root,model_name))
    return model_paths
def get_data_info(model_path):
    code,data=load_model(model_path)
    return data

def get_logp(model_path):
    code,data=load_model(model_path)
    model_name2 = re.sub('-', '_', model_path[1], flags=re.S) # replace dashes with underscores for internal use
    logp = logp_stan(code,data,model_name=model_name2)
    return logp

# iterator over all models
def all_logps():
    # iterator of the logps
    rez = map(get_logp,all_models())
    # filter to drop those that are None
    return filter(lambda a: a is not None, rez)

def load_data(model_path):
    # use rpy2 to interface with stan
    import readline # somehow makess rpy2 import correctly on swarm
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    from rpy2.robjects.vectors import FloatVector, IntVector
    import numpy as np
    # from rpy2.rinterface import RRuntimeError

    root,model_name = model_path

    ro.r('rm(list=ls())') # clear R memory
    try:
        ro.r.source('./' + root + '/' + model_name + '.data.R')
    except:# (RRuntimeError,RuntimeError):
        raise ValueError('error loading code of model ' + model_name)

    vars = list(ro.r.globalenv())

    data = dict()
    for var in vars:
        v0 = ro.r(var)
        v = np.array(v0)
        #print(type(ro.r(var)))
        #print(var,v.shape)
        #print(var, v.dtype)
        if not(v.dtype in [np.float64,np.int32]):
            raise ValueError('bad datatype in model ' + model_name)
        if all(v.ravel()-np.round(v.ravel())==0.0):
        #if type(v0)==IntVector:
            v = v.astype(int)
        # R represents scalars as Vectors-- need to detect this
        if (type(v0)==FloatVector or type(v0)==IntVector) and len(v)==1:
            v = v[0]
        data[var] = v
    return(data)

def load_code(model_path):
    root,model_name = model_path
    with open('./' + root + '/' + model_name + '.stan', 'r') as myfile:
        code = myfile.read().strip()
        return code

def load_model(model_path):
    return load_code(model_path), load_data(model_path)

# just check if the model can be loaded and a few inference routines can be done without problems
def is_good_model(model_path):
    # TODO : make sure data is deterministic
    try:
        logp   = get_logp(model_path)
        Z_nuts = logp.sampling(iter=20)
        Z_mf   = logp.mf(iter=100)
        Z_advi = logp.advi(iter=100)
        #print(z.shape,Z_nuts.shape,Z_mf.shape,Z_advi.shape)
        for Z in [Z_mf,Z_advi]:
            if Z.shape[1]!=Z_nuts.shape[1]:
                raise ValueError("NUTS and VI not same dimensions")

        # also check that optimization works
        z,rez  = logp.argmax(True,method='BFGS',gtol=1e-3,maxiter=10000) # a pretty relaxed view of optimization success
        if not rez.success:
            print(rez)
            raise ValueError("optimization failed")
        if len(z) != Z_nuts.shape[1]:
            raise ValueError("NUTS and argmax not same dimensions")

        return True

    except ValueError as e:
        print('got value error:',e)
        return False

###########################################################################
#  stuff related to compiling and caching stan models
###########################################################################

# removes whitespace and such to prevent extra compiling
# this is only used for hashing (original code is compiled)
def standardize_code(text):
    #text = re.sub('//.*?\n|/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//.*?\n|/\*.*?\*/', '', text, flags=re.S)
    pat = re.compile(r'\s+')
    text = pat.sub('', text)
    return text

import Cython
def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan` from pystan"""
    if not os.path.exists('./cached-models'):
        print('creating cached-models/ to save compiled stan models')
        os.makedirs('./cached-models')

    model_code_stripped = standardize_code(model_code)
    code_hash = md5(model_code_stripped.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = './cached-models/{}.pkl'.format(code_hash)
    else:
        cache_fn = './cached-models/{}-{}.pkl'.format(model_name, code_hash)

    if os.path.isfile(cache_fn):
        with open(cache_fn, 'rb') as f:
            sm = pickle.load(f)
    else:
        try:
            print("this is still recompiling models")            
            sm = pystan.StanModel(model_code=model_code,model_name=model_name,**kwargs)
        except Cython.Compiler.Errors.CompileError as e:
            raise ValueError('could not compile code for ' + model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)        
    return sm

###########################################################################
#  here's the object provided to the user-- a logp function that hooks into autograd
###########################################################################

class logp_stan:
    def __init__(self,model_code,data,model_name=None):
        extra_compile_args = ['-O1','-w','-Wno-deprecated'] # THIS IS SLOW! TURN OPTIMIZATION ON SOMEDAY!

        self.data = data
        self.model_name = model_name

        #self.sm = StanModel_cache(model_code=model_code,extra_compile_args=extra_compile_args,model_name=self.model_name)
        with suppress_stdout_stderr():
            warnings.filterwarnings("ignore")
            self.sm = StanModel_cache(model_code=model_code,extra_compile_args=extra_compile_args,model_name=self.model_name)
            warnings.filterwarnings("default")

        # self.fit = self.sm.sampling(data=self.data, iter=10, chains=1, init=0)
        # print(self.model_name)
        with suppress_stdout_stderr():
            warnings.filterwarnings("ignore")
            try:
                self.fit = self.sm.sampling(data=self.data, iter=10, chains=1, init=0)
            except RuntimeError as e:
                raise ValueError('could not init model for ' + model_code)
            warnings.filterwarnings("default")
        
        #self.ndims = len(self.fit.get_posterior_mean())
        self.zlen = len(self.fit.unconstrained_param_names())
        self.ndims = self.zlen

    def z0(self):
        return np.random.randn(self.zlen)

    def sampling(self,**kwargs):
        with suppress_stdout_stderr():
            warnings.simplefilter('ignore')
            self.fit = self.sm.sampling(data=self.data,**kwargs)
            rez = self.unconstrain(self.fit.extract())
            warnings.simplefilter('default')
            return rez

    # get posterior max via BFGS (or a method of your choice) -- passes keywords arguments to method
    def argmax(self,with_rez=False,method='BFGS',**kwargs):
        suggested_solvers = ['CG','BFGS','L-BFGS-B','TNC','SLSQP']
        if not method in suggested_solvers:
            warnings.warn("Solver" + str(method) +
            " passed to argmax not in suggested list" + str(suggested_solvers)+
            '\nsee list at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html')
        z = self.z0()
        obj = autograd.value_and_grad(lambda z : -self(z),0)
        kwargs['disp'] = False
        rez = scipy.optimize.minimize(obj,z,method=method,jac=True,options=kwargs)
        z = rez.x
        if with_rez:
            return rez.x, rez
        else:
            return rez.x

    # this works fine but isn't used...
    # def extract_flat(self):
    #     data = self.fit.extract()
    #     X = []
    #     for param_name in data:
    #         if param_name == 'lp__':
    #             continue
    #         myX = data[param_name]
    #         if myX.ndim > 1:
    #             myX = myX.reshape(myX.shape[0],np.prod(myX.shape[1:]))
    #         else:
    #             myX = myX.reshape(myX.shape[0],1)
    #         X.append(myX)
    #     return np.hstack(X)

    def deflat(self,params):
        data = self.fit.extract()
        where = 0
        stuff = []
        nsamps = params.shape[0]
        #print('data',data)
        for param_name in data:
            if param_name == 'lp__':
                continue
            print('param_name',param_name)
            myX = data[param_name]
            if myX.ndim > 1:
                howmany = np.prod(myX.shape[1:])
                print('myX.shape',myX.shape,'howmany',howmany,'params.shape',params.shape)
                t = params[:,where:where+howmany]
                where += howmany
                t = t.reshape(nsamps,*myX.shape[1:])
                stuff.append((param_name,t))
            else:
                t = params[:,where]
                where += 1
                stuff.append((param_name,t))
        return collections.OrderedDict(stuff)
        
    def advi(self,algorithm='fullrank',**kwargs):
        #algorithm='fullrank',iter=100000,tol_rel_obj=.0001
        try:
            with suppress_stdout_stderr():
                rez = self.sm.vb(data=self.data,algorithm=algorithm,**kwargs)
        except RuntimeError as e:
            print('ADVI failed with ', e)
            return np.array([[]])

        # get samples in messy ADVI format
        param_names = rez['sampler_param_names']
        samps       = np.array(rez['sampler_params'])[:-1,:].T

        # create nice format for target samples
        nsamps = samps.shape[0]
        nice_data = []
        for (param_name,mydata) in self.fit.extract().items():
            mysize = mydata.shape
            mysize = (nsamps,)+mysize[1:]
            #print('mysize',mysize)
            nice_data.append((param_name,np.zeros(mysize)))
        nice_data = collections.OrderedDict(nice_data)
        # cram the ADVI data in by parsing the freaking names
        for k in range(samps.shape[1]):
            pname = param_names[k]
            z     = samps[:,k]
            pname = str.split(pname,'.')
            #print('pname',pname)
            if len(pname)==1:
                n = pname[0]
                nice_data[n] = z
            else:
                n = pname[0]
                i = (Ellipsis,)+tuple(map(lambda x : int(x)-1,pname[1:]))
                nice_data[n][i] = z
        # now transform back to the original space (sigh...)
        return self.unconstrain(nice_data)
        
    # convenience wrapper for mean-field advi
    def mf(self,**kwargs):
        return self.advi(algorithm='meanfield',**kwargs)

    def constrain(self,Z):
        assert(Z.ndim==2)
        assert(Z.shape[1]==self.zlen)
        Y = np.array([self.fit.constrain_pars(np.asarray(z,order='C')) for z in Z])
        return Y

    # theta is a dict of parameters
    def unconstrain(self,thetas):
        #N = len(thetas['lp__'])
        N = list(thetas.values())[0].shape[0]
        #print('N=',N)
        W = []
        for n in range(N):
            theta = dict(((key,t[n]) for key,t in thetas.items()))
            W.append(self.fit.unconstrain_pars(theta))
        return np.array(W)

    @autograd.primitive
    def logp(self,z):
        assert(z.ndim==1)
        assert(len(z)==self.zlen)
        try: # try to evaluate logp with stan. if you fail, just return nan
            return self.fit.log_prob(z,adjust_transform=True)
        except ValueError:
            return np.nan
        return rez
        
    def __call__(self,z):
        return self.logp(z)

    def glogp(self,z):
        assert(len(z)==self.zlen)
        rez_from_stan = self.fit.grad_log_prob(z,adjust_transform=True)
        return rez_from_stan.reshape(z.shape)

    def vjpmaker(argnum,rez,stuff,args):
        obj = stuff[0]
        z   = stuff[1]
        if np.isnan(rez):
            return lambda gg : 0*z # special gradient for nan case
        else:
            return lambda gg : gg*logp_stan.glogp(obj,z)


autograd.extend.defvjp_argnum(logp_stan.logp, logp_stan.vjpmaker)

###########################################################################
#  a function to help Stan shut up
###########################################################################

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
