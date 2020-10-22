def optimize_single(hyper_params, uniq_name, perm_id):
    from pickle import load
    from json import dumps
    
    import mystan
    from inference import  run_iwvi
    import autograd.numpy.random as npr



    #load the relevant model
    with open('./good_model_paths.pkl','rb') as f:
        good_model_paths = load(f)
    del good_model_paths[9]

    model = mystan.get_logp(good_model_paths[hyper_params['good_model_id']])
    hyper_params['data_dim'] = model.zlen
    hyper_params['step_size'] = hyper_params['step_size']/hyper_params['data_dim']
    # using a fixed sample budget--we need to adjust the no. of training samples based on M 
    hyper_params['sample_size_training'] = (hyper_params['sample_memory_budget']//
                                            hyper_params['augmentation_param_M'])
    # Print basic model attributes
    print ("Running Gaussian VI on model :"+ model.model_name)
    print ("Model parameter dimensions : ", model.zlen)
    
    with open("../ezstan/good_model_data_length.pkl", "rb") as f: 
        good_model_data_length =  load(f)
   
    print ("Model data size : ", int(good_model_data_length[hyper_params['good_model_id']]))



    #set up the seed and call the inference function 
    npr.seed(hyper_params['seed'])

    # Print the entire hyperparam dictionary
    [print(f"{k} : {i}") for k, i in hyper_params.items()];

    run_iwvi(model = model, hyper_params = hyper_params, uniq_name = uniq_name)



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
        help="Id of the models to experiment with. Should use Ids from table 1 in the paper.", nargs = "*") 

    args = parser.parse_args()
    uniq_name = args.uname
    models = args.models

    with open("../ezstan/good_model_new_id_to_old_id_dict.pkl", "rb") as f:
        good_model_orig_id_dict = load(f)

    for m in models:
        hyper_params['good_model_id'] = good_model_orig_id_dict[m]
        for step in hyper_params['step_size_range']:
            hyper_params['step_size'] = step
            try :
                start = time.time()
                optimize_single(hyper_params = hyper_params.copy(), uniq_name = uniq_name, perm_id = 0)
                print("The time taken for the process : ",time.time() - start)
            except Exception as e:
                print(f"Error during the training of model {m}...")
                print(e)

    save_current_dict(hyper_params = hyper_params, uniq_name = uniq_name)
