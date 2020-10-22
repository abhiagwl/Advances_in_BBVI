import time
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from vi_families import gaussian_sample, gaussian_entropy, gaussian_logpdf, rnvp_sample_logpdf, rnvp_inverse_sample_logpdf, softmax, softmax_matrix
from autograd.scipy.special import logsumexp
from autograd import value_and_grad
from scipy.optimize import minimize
from autograd.misc import flatten
from math import isnan
from autograd.core import getval
import time 
from utilities.helper import evaluate_batch_logp, coupling_layer_specifications
from utilities.result_helper import save_results_parameters, store_laplaces_init, retrieve_stored_laplaces_init, check_laplaces_init_stored
from inference import init_params_full_cov_gaussian, laplace_initialization, init_params_rnvp
from collections import deque

def advi_optimizer(grad, x, callback, num_iters, eta, epsilon = 1e-16, tau = 1, alpha = 0.1):
    x, unflatten = flatten(x)
    s = np.zeros(len(x))

    for i in range(num_iters):
        g = flatten(grad(unflatten(x), i))[0]
        if callback: callback(unflatten(x), i, unflatten(g))
        if i==0:
            s = g**2
        else:
            s = alpha*(g**2) + (1-alpha)*s
        eta_scaled = eta / np.sqrt(i+1)
        x = x -  eta_scaled*g/ (tau + np.sqrt(s))

    return unflatten(x)

def advi_baseline_asserts(hyper_params):

    assert(hyper_params['advi_use'] == 1)
    assert(hyper_params['advi_use_baseline'] == 1)
    assert(hyper_params['vi_family'] == 'gaussian')
    assert(hyper_params['augmentation_param_M'] == 1)
    assert(hyper_params['laplaces_method_use'] == 0)
    assert(hyper_params['grad_estimator_type'] == 'closed-form-entropy')

def run_advi(model, hyper_params, uniq_name, run_adapt_eta):#vi_family, k = 1):

    def run_optimization(obj, obj_grad, init_params, eta, num_iters, callback = None):
        start_time = time.time()  
        optimized_params = advi_optimizer(obj_grad, init_params, eta=eta,
                            num_iters=num_iters, callback=callback)
        print ("Time just for optimization: ", time.time()- start_time)
        return optimized_params


    def advi_elbo(params, sample_size_elbo_evaluation):

        samples = gaussian_sample(mu = params[0], sig = np.tril(params[1]), 
            sample_size = (sample_size_elbo_evaluation, hyper_params['augmentation_param_M']))

        cll = []
        for i in range(sample_size_elbo_evaluation):
            cll.append(evaluate_batch_logp(model, samples[i]))
        cll = np.array(cll)

        assert(cll.shape == (sample_size_elbo_evaluation, 1))
        assert(cll.shape == samples.shape[:-1])

        cll = cll.reshape((sample_size_elbo_evaluation,))
        hq  = gaussian_entropy(mu = params[0], sig = np.tril(params[1]))
        assert(not isinstance(hq, np.ndarray))

        elbo = np.mean(cll) + hq 
        del cll, samples

        return elbo

    def advi_objective(params, t):
        return -1.0*advi_elbo(params, hyper_params['sample_size_training'])
    
    ##################################################################
    # 
    #     for each eta
    #         we run the optimization with the adapt parameters
    #         we collect the best values and then use that for comparison
    ##################################################################
    def adapt_eta():

        init_elbo = objective_to_evaluate(params = init_params, sample_size_elbo_evaluation = hyper_params['advi_adapt_eta_evaluation_sample_size'])
        best_elbo = -1.0*np.inf
        best_eta = 0

        if hyper_params['verbose'] == 1: 
            print("####################### ")
            print("Initial elbo: ", init_elbo)
            print("####################### ")

        for i, eta in enumerate(hyper_params['advi_adapt_eta_range']):
            results = []
            hyper_params['advi_eta'] = eta

            if hyper_params['verbose'] == 1: 
                print("####################### ")
                print("Checking the eta: ", eta)
                print("####################### ")
            try: 
                optimized_params = run_optimization(obj = objective_to_opt, obj_grad = objective_to_opt_grad, init_params = init_params, 
                    eta = eta, num_iters = hyper_params['advi_adapt_eta_num_iters'])
            except Exception as e:
                print(f"Error occured during the optimization with step-size {eta}...")
                print(e)
                print(f"Using initial parameters instead for {eta}...")
                optimized_params = init_params

            candidate_elbo  = objective_to_evaluate(params = optimized_params, sample_size_elbo_evaluation = hyper_params['advi_adapt_eta_evaluation_sample_size'])
            candidate_elbo = -1.0*np.inf if np.isnan(candidate_elbo) else candidate_elbo 
            
            if ((candidate_elbo < best_elbo) & (best_elbo > init_elbo)):
                assert(best_eta!= 0)
                if hyper_params['verbose'] == 1: 
                    print("Best eta found, best eta : ", best_eta)
                    print("Best eta found, best elbo : ", best_elbo)
                return best_eta
            else:
                if((i+1) < len(hyper_params['advi_adapt_eta_range'])):
                    best_elbo = candidate_elbo
                    best_eta = eta
                else:
                    if candidate_elbo > init_elbo:
                        if hyper_params['verbose'] == 1: 
                            print("Best eta found, best eta : ", best_eta)
                            print("Best eta found, best elbo : ", best_elbo)
                        return best_eta
                    else :
                        print("ELBO value diverged for all etas. Update eta range")
                        exit()

    def run_with_eta(eta):

        def relative_difference(curr, prev):
            return np.abs((curr-prev)/prev)

        def full_run_callback(params, t, g):

            results.append(objective_to_evaluate(params, hyper_params['sample_size_training']))
            if (t+1)%hyper_params['call_back_iteration']==0:
                print("Iteration {} log likelihood LOWER BOUND(RUNNING AVERAGE) {}".format(t+1, np.nanmean(results)))
                print("Iteration {} log likelihood LOWER BOUND(CURRENT ESTIMATE) {}".format(t+1, results[-1]))

                previous_elbo = results[-(hyper_params['call_back_iteration']+1)] if len(results) > hyper_params['call_back_iteration'] else 0.0 
                current_elbo = results[-1]
                delta_results.append(relative_difference(previous_elbo, current_elbo))

                delta_elbo_mean = np.nanmean(delta_results)
                delta_elbo_median = np.nanmedian(delta_results)

                print("Iteration {} delta mean {}".format(t+1, delta_elbo_mean))
                print("Iteration {} delta median {}".format(t+1, delta_elbo_median))

                if delta_elbo_median <= hyper_params['advi_convergence_threshold']:
                    print("converged according to ADVI metrics for Median")
                    tn = time.time() - t0
                    save_results_parameters(hyper_params =hyper_params,
                     params = params,
                     model = model,
                     uniq_name = uniq_name + str("_delta_convergence_"),
                     results = results,
                     time = tn/(t+1))
                    exit()
                elif delta_elbo_mean <= hyper_params['advi_convergence_threshold']:
                    print("converged according to ADVI metrics for Mean")
                    tn = time.time() - t0
                    save_results_parameters(hyper_params =hyper_params,
                     params = params,
                     model = model,
                     uniq_name = uniq_name + str("_delta_convergence_"),
                     results = results,
                     time = tn/(t+1))
                    exit()

            if hyper_params['check_point_use']==1:
                if (t+1) in hyper_params['check_point_num_epochs']:
                    tn = time.time() - t0
                    save_results_parameters(hyper_params =hyper_params,
                     params = params,
                     model = model,
                     uniq_name = uniq_name+"_check_point_"+str(t+1),
                     results = results,
                     time = tn/(t+1))

        buffer_len = np.int(max(0.01*hyper_params['num_epochs']/hyper_params['call_back_iteration'] , 2))
        print("buffer_len : ", buffer_len)
        delta_results = deque(maxlen = buffer_len)
        current_elbo = 0.0
        results = []
        t0 = time.time()
        optimized_params = run_optimization(obj = objective_to_opt, obj_grad = objective_to_opt_grad, init_params = init_params, eta = eta, num_iters = hyper_params['num_epochs'], callback = full_run_callback)
        if hyper_params['check_point_use'] == 0:
            tn = time.time() - t0
            save_results_parameters(hyper_params =hyper_params,
             params = optimized_params,
             model = model,
             uniq_name = uniq_name,
             results = results,
             time = tn/(hyper_params['num_epochs']))

    advi_baseline_asserts(hyper_params)

    init_params = init_params_full_cov_gaussian(D = hyper_params['data_dim'])

    objective_to_opt = advi_objective

    objective_to_evaluate = advi_elbo

    objective_to_opt_grad = grad(objective_to_opt) 

    if run_adapt_eta == 1:
        best_eta = adapt_eta()
        run_with_eta(eta = best_eta)
    else :
        run_with_eta(eta = hyper_params['advi_eta'])

