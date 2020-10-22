import time
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
from vi_families import (GaussianVI, gaussian_sample, gaussian_entropy, 
                         mix_of_gaussian_logpdf, gaussian_logpdf, rnvp_sample_logpdf, 
                         rnvp_inverse_sample_logpdf, generate_mask, softmax, softmax_matrix, 
                         forward_transform)
# from autograd.scipy.special import logsumexp
from autograd.scipy.special import logsumexp
from autograd import value_and_grad
from scipy.optimize import minimize
from autograd.misc import flatten
from math import isnan
# from config import hyper_params 
from autograd.core import getval
import time 
from utilities.helper import (evaluate_batch_logp, coupling_layer_specifications,
                              good_iter)
from utilities.result_helper import (save_results_parameters, 
                                     store_laplaces_init, retrieve_stored_laplaces_init, 
                                     check_laplaces_init_stored)

def init_params_full_cov_gaussian(D):
    """returns the mean and cholesky factor of the full cov gaussian distribution"""
    return [np.zeros(D,), np.eye(D)]

def init_params_diag_cov_gaussian(D):
    """returns the mean and cholesky factor of the diag gaussian distribution"""
    return [np.zeros(D,), np.zeros(D,)]

def laplaces_init_params_full_cov_gaussian(model, hyper_params):        
    # Initialize both the parameters using the solution of MAP using some BFGS type method
    # basically to initialize to solution of the laplace's method 
    D = model.zlen
    z_0 = npr.rand(D)
    obj = value_and_grad(lambda z: -model.logp(z))
    rez = minimize(obj, z_0, method='BFGS', jac=True, options={'maxiter':hyper_params['laplaces_method_map_estimate_num_epochs'], 'disp':True})
    mu = rez.x

    def finite_differences(z, grad_f, epsilon = hyper_params['laplaces_method_epsilon']):
        D = len(z)
        H = np.zeros((D,D))
        for d in range(D):
            z_pos = z*1.0
            z_pos[d] += epsilon
            z_neg = z*1.0
            z_neg[d] -= epsilon

            H[:,d] = (grad_f(z_pos) - grad_f(z_neg))/(2*epsilon)
        return H
    H = finite_differences(z = mu, grad_f = grad(lambda z : -model.logp(z)), epsilon = 1e-4)
    neg_H_inv_3 = np.linalg.inv(H)
    try :
        L = np.linalg.cholesky(neg_H_inv_3) # -H_inv = inv(-H)
    except:
        print ('Using unit normal for generating random cholesky factor')
        L = np.tril(npr.randn(D,D))
    return mu, L

def laplace_initialization(hyper_params, model ):
    if check_laplaces_init_stored(hyper_params = hyper_params):
        init_params = retrieve_stored_laplaces_init(hyper_params = hyper_params)
    else :
        init_params = laplaces_init_params_full_cov_gaussian(model = model, hyper_params = hyper_params)
        store_laplaces_init(params = init_params , hyper_params = hyper_params)
    return init_params



def init_params_rnvp(hyper_params):
    def generate_net_st():
        coupling_layer_sizes = coupling_layer_specifications(hyper_params = hyper_params)
        init_st_params = []
        for layer_sizes in coupling_layer_sizes:
             init_st_params.append([(hyper_params['params_init_scale'] * npr.randn(m, n),   # weight matrix
                                           hyper_params['params_init_scale'] * npr.randn(n))      # bias vector
                                          for m, n in zip(layer_sizes[:-1], layer_sizes[1:])])
        return init_st_params
    st = [generate_net_st() for i in range(hyper_params['rnvp_num_transformations'])]
    return st

def print_init_param_shape(init_params):
    print("here")
    st = init_params[0]
    for coupling_layers in st:
        for layer_sizes in coupling_layers:
            print(layer_sizes[0].shape)
            print(layer_sizes[1].shape)

def objective_weights(lp, lq_stopped, hyper_params):
    lw = getval(lp - lq_stopped)
    w = softmax_matrix(lw)
    if hyper_params['grad_estimator_type']== 'IWAE':
        return w
    elif hyper_params['grad_estimator_type'] == "IWAESTL": 
        return w
    elif hyper_params['grad_estimator_type'] == "IWAEDREG": 
        return np.power(w,2)

    else :
        print ("not implemented")
        exit()


def objective_target(lp, lq, lq_stopped, hyper_params):
    if hyper_params['grad_estimator_type']== 'IWAE':
        return lp - lq

    elif hyper_params['grad_estimator_type'] == "IWAESTL": 
        return lp - lq_stopped
    
    elif hyper_params['grad_estimator_type'] == "IWAEDREG": 
        return lp - lq_stopped

    else :
        print ("not implemented")
        exit()

def diag_sqrt_exp(x):
    return np.diag(np.sqrt(np.exp(x))) 

def params_to_mu_sig(params, hyper_params):
    if hyper_params['vi_family'] == "gaussian - diagonal":
        mu_, sig_ = params[0], diag_sqrt_exp(params[1])
    else:
        mu_, sig_ = params[0], np.tril(params[1])
    return mu_, sig_

def cll_evaluation(model, samples, n_samples):
    cll_ = []
    for i in range(n_samples):
        cll_.append(evaluate_batch_logp(model, samples[i]))
    return np.array(cll_)

def run_iwvi(model, hyper_params, uniq_name):#vi_family, k = 1):
    def run_optimization(obj, obj_grad, init_params, hyper_params, callback):
        start_time = time.time()  
        optimized_params = adam(obj_grad, init_params, step_size=hyper_params['step_size'],
                            num_iters=hyper_params['num_epochs'], callback=callback)
        print ("Time just for optimization: ", time.time()- start_time)
        return optimized_params


    def iwelbo(params):
        params = (params)
        if 'gaussian' in hyper_params['vi_family']:
            mu_, sig_ = params_to_mu_sig(params, hyper_params)
            samples = gaussian_sample(mu = mu_, sig = sig_, 
                sample_size = (hyper_params['sample_size_training'],hyper_params['augmentation_param_M']))
            lq = gaussian_logpdf(samples = samples, mu = mu_, sig = sig_)

        elif hyper_params['vi_family'] == 'rnvp':
            samples, lq = rnvp_sample_logpdf(params = params, hyper_params = hyper_params,
                sample_size = (hyper_params['sample_size_training'],hyper_params['augmentation_param_M']))
        cll = cll_evaluation(model, samples, hyper_params['sample_size_training'])

        assert(cll.shape == lq.shape)
        assert(cll.shape == samples.shape[:-1])

        iw_elbo = -1.0*np.log(hyper_params['augmentation_param_M']) + logsumexp(cll - lq, -1)
        del cll, lq, samples
        assert(iw_elbo.shape == (hyper_params['sample_size_training'],))

        return np.mean(iw_elbo)


    def objective_gaussian(params,t):
        '''
        generates a random smaple batch from the q distribution and evaluates the ELBO on it
        '''
        mu_, sig_ = params_to_mu_sig(params, hyper_params)

        samples = gaussian_sample(mu = mu_, sig = sig_, 
            sample_size = (hyper_params['sample_size_training'],hyper_params['augmentation_param_M']))
        #stan currently doesnot support batching of inputs for logpdf calculation
        #iterative scheme to calculate for a batch
        cll = cll_evaluation(model, samples, hyper_params['sample_size_training'])

        assert(cll.shape == (hyper_params['sample_size_training'],hyper_params['augmentation_param_M']))


        if (hyper_params['augmentation_param_M'] == 1) & (hyper_params['grad_estimator_type'] == "closed-form-entropy"):
            hq  = gaussian_entropy(mu = params[0], sig = np.tril(params[1]))
            assert(not isinstance(hq, np.ndarray))
            elbo = np.mean(cll) + hq 
            del cll, samples
            return -1.0*elbo

        else:
            params_stopped = getval(params)

            mu_stopped, sig_stopped = params_to_mu_sig(params_stopped, hyper_params)

            lq_stopped = gaussian_logpdf(samples = samples, mu = mu_stopped, sig = sig_stopped) 
            lq = gaussian_logpdf(samples = samples, mu = mu_, sig = sig_)

            assert(cll.shape == lq.shape)

            weights = objective_weights(lp = cll, lq_stopped = lq_stopped, hyper_params = hyper_params)
            targets = objective_target(lp = cll, lq = lq, lq_stopped = lq_stopped, hyper_params = hyper_params)

            assert(weights.shape == targets.shape)

            final_target = np.sum(np.multiply(weights,targets), -1)
            assert(final_target.shape == (hyper_params['sample_size_training'], ))
            return -1.0*np.mean(final_target)




    def objective_rnvp(params,t):
        '''
        generates a random smaple batch from the q distribution and evaluates the ELBO on it
        '''

        samples, lq = rnvp_sample_logpdf(params = params, hyper_params = hyper_params,
            sample_size = (hyper_params['sample_size_training'],hyper_params['augmentation_param_M']))



        params_stopped = getval(params)
        _, lq_stopped = rnvp_inverse_sample_logpdf(params = params_stopped, hyper_params = hyper_params, samples = samples) 


        #stan currently doesnot support batching of inputs for logpdf calculation
        #iterative scheme to calculate for a batch
        cll = cll_evaluation(model, samples, hyper_params['sample_size_training'])

        assert(cll.shape == (hyper_params['sample_size_training'],hyper_params['augmentation_param_M']))
        assert(cll.shape == lq.shape)
        
        weights = objective_weights(lp = cll, lq_stopped = lq_stopped, hyper_params = hyper_params)
        targets = objective_target(lp = cll, lq = lq, lq_stopped = lq_stopped, hyper_params = hyper_params)

        assert(weights.shape == targets.shape)

        final_target = np.sum(np.multiply(weights,targets), -1)
        assert(final_target.shape == (hyper_params['sample_size_training'], ))
        return -1.0*np.mean(final_target)
        



    def callback(params, t, g):
        results.append(iwelbo(params))
        if good_iter(t+1):
            if np.isnan(results[-1]) : 
                print("exiting optimization because nan encountered.")
                raise ValueError
            print("Iteration {} log likelihood LOWER BOUND(RUNNING AVERAGE) {}".format(t+1, np.mean(results)))
            print("Iteration {} log likelihood LOWER BOUND(CURRENT ESTIMATE) {}".format(t+1, results[-1]))
        if hyper_params['check_point_use']==1:
            if (t+1) in hyper_params['check_point_num_epochs']:
                tn = time.time() - t0
                save_results_parameters(hyper_params =hyper_params,
                 params = params,
                 model = model,
                 uniq_name = uniq_name+"_check_point_"+str(t+1),
                 results = results,
                 time = tn/(t+1)
                 )
    

    results = []

    if "gaussian" in hyper_params['vi_family']:
        if hyper_params['vi_family'] == "gaussian":
            if hyper_params['laplaces_method_use'] == 1:
                init_params = laplace_initialization(hyper_params = hyper_params, model = model)
            else:
                init_params = init_params_full_cov_gaussian(D = hyper_params['data_dim'])
        elif hyper_params['vi_family'] == "gaussian - diagonal":
            init_params = init_params_diag_cov_gaussian(D = hyper_params['data_dim'])
        else : raise ValueError

        objective_to_opt = objective_gaussian       


    elif hyper_params['vi_family'] =="rnvp":
        init_params = init_params_rnvp(hyper_params)
        if hyper_params['laplaces_method_use'] == 1: raise ValueError
            # laplaces_method_mu, laplaces_method_L = laplace_initialization(hyper_params = hyper_params, model = model)
            # hyper_params['laplaces_method_mu'] = laplaces_method_mu
            # hyper_params['laplaces_method_L'] = laplaces_method_L
          
        objective_to_opt = objective_rnvp

    else:
        raise ValueError ("VI family not supported yet")
        exit()
    objective_to_opt_grad = grad(objective_to_opt) 
    
    t0 = time.time()
    
    optimized_params = run_optimization(
        obj = objective_to_opt,
         obj_grad = objective_to_opt_grad,
         init_params = init_params,
         hyper_params= hyper_params,
         callback = callback
     )

    if hyper_params['check_point_use'] == 0:
        tn = time.time() - t0
        save_results_parameters(hyper_params =hyper_params,
         params = optimized_params,
         model = model,
         uniq_name = uniq_name,
         results = results,
         time = tn/hyper_params['num_epochs'])






def iw_posterior_sample(num_samples, hyper_params, params, model):
    if hyper_params['vi_family']=='rnvp' :
        samples, lq = rnvp_sample_logpdf(params = params, hyper_params = hyper_params,
            sample_size = (num_samples, hyper_params['augmentation_param_M']))
    elif 'gaussian' in hyper_params['vi_family']:
        mu_, sig_ = params_to_mu_sig(params, hyper_params)
        samples = gaussian_sample(mu = mu_, sig = sig_, 
            sample_size = (num_samples, hyper_params['augmentation_param_M']))
        lq = gaussian_logpdf(samples = samples, mu = mu_, sig = sig_)
    else :
        raise ValueError ("VI family not supported yet")
        exit()

    if hyper_params['mog_use']==1:
        cll = mix_of_gaussian_logpdf(samples = samples, normals = hyper_params['mog_normals'], a = hyper_params['mog_a'])
    else:
        cll = cll_evaluation(model, samples, num_samples)

    lR = cll - lq
    final_samples = []
    for i in range(lR.shape[0]):
        j = np.argmax(npr.multinomial(1, softmax(lR[i])))
        final_samples.append(samples[i,j,:])
    return np.array(final_samples)



