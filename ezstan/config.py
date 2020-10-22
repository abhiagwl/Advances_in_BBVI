import autograd.numpy as np 
import itertools
from pickle import load
#hyper-parameter default choices
hyper_params = {
	# step-sizes are scaled with dimensionality of the problem later
	'step_size' : 0.1/4**2,
	'step_size_range' : [ .1, .1/4, .1/4**2, .1/4**3, .1/4**4], 
	# works for both ADVI and our VI methods
	'num_epochs' : 20,
	# works for both ADVI and our VI methods
	'sample_size_training' : 1, # changed later based on the sample_memory_budget
	# works for both ADVI and our VI methods
	'seed' : 11,
	# works for both ADVI and our VI methods
	'good_model_id'  : 128,
	# Choices for vi_families: 'rnvp', 'gaussian', 'gaussian - diagonal'
	'vi_family' : "rnvp",
	}
hyper_params.update({
	# augmentation_param_M is same as M_training 
	# M_training = 1 corresponds to regular VI training
	# M_training > 1 corresponds to IW-training
	# Try 1 or 10; other values will need corresponding adjustment in evaluation 
	'augmentation_param_M'  : 10, 
	})

hyper_params.update({
	# Three choices for gradient estimator type: 'IWAE', 'IWAEDREG', 'closed-form-entropy'
	# closed-form-entropy should only be used with real-NVP
	# IWAEDREG defaults to STL when augmentation_param_M = 1  
        'grad_estimator_type': 'IWAEDREG', 
	})
hyper_params.update({
    # M_sampling : change here if you try different value of M_training
    'sampling_M_use' : 1,
    'sampling_M_range' : [1, 10], 
	})
hyper_params.update({
	# to select the real-nvp number fo transformation and hidden layer size; to change the number of hidden layer, see vi_families.py
	'rnvp_num_transformations' : 10,
	'rnvp_num_hidden_units' : 32,
	'params_init_scale' : 0.001
})
hyper_params.update({
	# used for final metric evaluations
	'sample_size_evaluation' : 10000 if hyper_params['num_epochs'] >10 else hyper_params['num_epochs'], 
	# used to control the overall samples used for final metric evaluations
	'fix_evaluation_budget': 1, 
	# used for training; fixes training time oracle evaluations
	'sample_memory_budget':100, 
	})

hyper_params.update({
	# Change here to use LI
	# Do not use LI with real-NVP
	'laplaces_method_use':0,
	'laplaces_method_map_estimate_num_epochs':2000,
	'laplaces_method_epsilon':1e-6,
	})

hyper_params.update({
	'check_point_use': 0,
})
