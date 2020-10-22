
def optimize_single(hyper_params, uniq_name, perm_id):
    from pickle import load
    
    import mystan
    from advi_inference import  run_advi
    import autograd.numpy.random as npr

    #load the relevant model
    with open('./good_model_paths.pkl','rb') as f:
        good_model_paths = load(f)
    del good_model_paths[9]

    model = mystan.get_logp(good_model_paths[hyper_params['good_model_id']])
    hyper_params['data_dim'] = model.zlen

    hyper_params['step_size'] = hyper_params['step_size']/hyper_params['data_dim']
    hyper_params['sample_size_training'] = hyper_params['sample_memory_budget']//hyper_params['augmentation_param_M']
    assert(hyper_params['sample_memory_budget']>=hyper_params['augmentation_param_M'])
        

    
    print ("Running Gaussian VI on model :"+ model.model_name)
    print ("Model parameter dimensions : ", model.zlen)
    with open("../ezstan/good_model_data_length.pkl", "rb") as f: 
        good_model_data_length =  load(f)
    print ("Model data size : ", 
        int(good_model_data_length[hyper_params['good_model_id']]))

    #set up the seed and call the inference function 
    npr.seed(hyper_params['seed'])
    [print(f"{k} : {i}") for k, i in hyper_params.items()];

    adapt_eta = 1 if hyper_params['advi_adapt_eta_use'] == 1 else 0 
    run_advi(model = model, hyper_params = hyper_params, 
        uniq_name = uniq_name, run_adapt_eta = adapt_eta)





if __name__ == "__main__" :

    import argparse
    import time
    from config import hyper_params
    from pickle import load
    from dict_saver import save_current_dict

    parser = argparse.ArgumentParser()
    parser.add_argument('-uname', '--uname', type=str, required=True, 
        help="Unique name used for saving and retrieving.") 
    parser.add_argument('-models', '--models', type=int, required=True, 
        help="Id of the models to experiment with. Should use the Ids from table 1 in the paper.", nargs = "*") 

    args = parser.parse_args()
    uniq_name = args.uname
    models = args.models
    hyper_params['models_experimented'] = models

    with open("../ezstan/good_model_new_id_to_old_id_dict.pkl", "rb") as f:
        good_model_orig_id_dict = load(f)

    hyper_params.update({
    'verbose':1, # for detailed message for ADVI run
    # explicitly set the hyper-params for the ADVI run. This overwrites the existing hyper-params in the config.py
    'vi_family':"gaussian",
    'augmentation_param_M':1,
    'laplaces_method_use':0,
    'grad_estimator_type':'closed-form-entropy',
    'step_size':-1,

    'advi_use' : 1, # overall using advi steps or not
    'advi_use_baseline' : 1, # using advi with baseline setting
    'advi_eta': 0.1, # the eta that is used with the optimizer if adapt eta is not used 
    'advi_convergence_threshold' : 0.001,
    'advi_adapt_eta_use': 1, # flag to indicate whether to use adaptive eta search or not
    'advi_adapt_eta_range': [100,10,1,0.1,0.01],
    'advi_adapt_eta_num_iters': 20, # used for the number of updates for each different eta in eta range
    'advi_adapt_eta_evaluation_sample_size' : 500, # used for comparing final elbo values
    
    'call_back_iteration': 500 if hyper_params['num_epochs']==30000 else int(hyper_params['num_epochs']//10),
    })

    for m in models:
        hyper_params['good_model_id'] = good_model_orig_id_dict[m]
        try :
            start = time.time()
            optimize_single(hyper_params = hyper_params.copy(), uniq_name = uniq_name, perm_id = 0)
            print("The time taken for the process : ",time.time() - start)
        except Exception as e:
            print(f"Error during the training of model {m}...")
            print(e)

    save_current_dict(hyper_params = hyper_params, uniq_name = uniq_name)
