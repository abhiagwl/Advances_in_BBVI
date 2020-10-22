import autograd.numpy as np 
import itertools
import os
import pickle
import pandas as pd
import argparse
from utilities.helper import (dict_to_result_dir, result_dir_to_file,  dict_to_dataframe) 
from utilities.result_helper import save_dataframe
from time import sleep


def retrieve_single(hyper_params, uniq_name, data):
    with open('./good_model_paths.pkl','rb') as f:
        good_model_paths = load(f)
    del good_model_paths[9]
    # print((good_model_paths[hyper_params['good_model_id']]))
    model = mystan.get_logp(good_model_paths[hyper_params['good_model_id']])
    hyper_params['data_dim'] = model.zlen
    # if perm_id == -1: 
    hyper_params['step_size'] = hyper_params['step_size']/hyper_params['data_dim']
    hyper_params['sample_size_training'] = hyper_params['sample_memory_budget']//hyper_params['augmentation_param_M']

    directory_name = dict_to_result_dir(hyper_params = hyper_params)

    if hyper_params['check_point_use'] == 0:
        result_file_name = result_dir_to_file(directory_name = directory_name, 
            output = 'results', uniq_name = uniq_name, hyper_params = hyper_params)
        if os.path.exists(result_file_name):
            with  open(result_file_name, 'rb') as f:
                results, time = pickle.load(f)
            data.loc[len(data)] = dict_to_dataframe(hyper_params = hyper_params, 
                results = results, is_seed =0) + [time]
    else:
        raise ValueError

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-uname', '--uname', type=str, required=True) 
    args = parser.parse_args()
    uniq_name = args.uname

    from dict_saver import retrieve_u_name_dict_and_update 
    retrieve_u_name_dict_and_update(uniq_name = uniq_name)
    sleep(0.5)
    from config import hyper_params

    data = pd.DataFrame(columns = utilities.helper.result_dir_hparams + ['training_average_elbo']+ ['time per iter'])

    for m in hyper_params['models_experimented']:
        # change the relevant entries in hyper_params dictionary for easy retrieval
        hyper_params['good_model_id'] = m
        data = retrieve_single(hyper_params = hyper_params.copy(), uniq_name = uniq_name, data = data)


    print (data.head())
    print (data.shape)

    save_dataframe(dataframe = data, hyper_params = hyper_params, uniq_name = str(uniq_name))
