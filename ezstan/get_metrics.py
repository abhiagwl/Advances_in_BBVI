import autograd.numpy as np 
import itertools
import os
from pickle import load
import pandas as pd
import argparse
from utilities.helper import (dict_to_result_dir, result_dir_to_file,  dict_to_dataframe, result_dir_hparams, dump_pickled_files, open_pickled_files) 
from utilities.result_helper import save_dataframe
from time import sleep


def retrieve_single(hyper_params, uniq_name, data):
    import mystan
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
                results, time = load(f)
            data.loc[len(data)] = dict_to_dataframe(hyper_params = hyper_params, 
                results = results, is_seed =0) + [time]
    else:
        raise ValueError

    return data

def get_training_data(hyper_params, uniq_name):
    print("Gathering training data...")

    data = pd.DataFrame(columns = result_dir_hparams + ['training_average_lower_bound']+ ['time per iter'])
    # print(hyper_params['models_experimented'])
    with open("../ezstan/good_model_new_id_to_old_id_dict.pkl", "rb") as f:
        good_model_orig_id_dict = load(f)

    for m in hyper_params['models_experimented']:
        hyper_params['good_model_id'] = good_model_orig_id_dict[m]
        if hyper_params['step_size'] == -1:
            data = retrieve_single(hyper_params = hyper_params.copy(), uniq_name = uniq_name, data = data)
        else:
            for step in hyper_params['step_size_range']:
                hyper_params['step_size'] = step
                data = retrieve_single(hyper_params = hyper_params.copy(), uniq_name = uniq_name, data = data)


    if hyper_params['step_size'] == -1:
        data['step_size_scheme'] = 'ADVI'
    else :
        data['step_size_scheme'] = 'comprehensive step search'

    print (data.head(20))
    # print (data.shape)

    save_dataframe(dataframe = data, hyper_params = hyper_params, uniq_name = str(uniq_name))
    return data

def get_clean_training_data(data):
    print("\n\nCleaning training dataframe before final metric calculation; selects best q based on average training ELBO from across the step-sizes in comprehensive search...")
    with open("../ezstan/good_model_name.pkl", 'rb') as f:
         good_model_names = load(f)

    data['good_model_names'] = data['good_model_id'].apply(lambda x: good_model_names[np.int(x)])
    data['good_model_id'] = data['good_model_id'].astype('int')
    data['num_epochs'] = data['num_epochs'].astype('int')
    data['step_size'] = data['step_size'].astype('float')
    
    to_observe = [ 'vi_family', 'good_model_id', 'augmentation_param_M','laplaces_method_use', 'grad_estimator_type', 'seed', 'num_epochs']
    to_pivot = 'training_average_lower_bound'
    idx = data.groupby(to_observe)[to_pivot].transform(np.nanmax)==data[to_pivot]
    data = data[idx].sort_values(by =  ['good_model_id','vi_family'])
    
    data['augmentation_param_M'] = data['augmentation_param_M'].apply(lambda x : np.int(x))
    print("Cleaned Data...")
    print(data.head(20))

    return data



def metric_iwelbo_eval(params, model, hyper_params):
    from vi_families import gaussian_sample, gaussian_logpdf, rnvp_sample_logpdf
    from utilities.helper import evaluate_batch_logp
    from autograd.scipy.special import logsumexp

    evaluation_sample_size = hyper_params['sample_size_evaluation']

    if hyper_params['vi_family'] == 'gaussian':
        samples = gaussian_sample(mu = params[0], sig = np.tril(params[1]), 
            sample_size = (evaluation_sample_size,))
        lq = gaussian_logpdf(samples = samples, mu = params[0], sig = np.tril(params[1]))
    
    elif hyper_params['vi_family'] == 'rnvp':
        samples, lq = rnvp_sample_logpdf(params = params, hyper_params = hyper_params,
            sample_size = (evaluation_sample_size, ))
    
    else : raise NotImplementedError

    assert(len(samples.shape) == 2)

    cll = evaluate_batch_logp(model, samples)
    assert(cll.shape == lq.shape)
    assert(cll.shape == samples.shape[:-1])

    metrics = []
    metrics_std = []
    for m in hyper_params['sampling_M_range']:
        batch_size = evaluation_sample_size//m
        assert(batch_size*m == evaluation_sample_size)
        iw_elbo = (-1.0*np.log(m) + logsumexp(cll.reshape(batch_size, m) 
                                            - lq.reshape(batch_size, m), -1))
        assert(iw_elbo.shape == (batch_size,))
        metrics.append(np.mean(iw_elbo))
        metrics_std.append(np.std(iw_elbo))

    return metrics, metrics_std, samples


def get_final_metric_data(data, hyper_params, uniq_name):
    print("\n\nCalculating final metric, and collecting samples from the final q distributions")
    import mystan
    metrics_list = []
    metrics_std_list = []

    for i, row in data.iterrows():
        hyper_params['good_model_id'] = row['good_model_id']
        hyper_params['step_size'] = row['step_size']

        with open("../ezstan/good_model_length.pkl", 'rb') as f:
            good_model_zlens = load(f)
        hyper_params['data_dim'] = good_model_zlens[hyper_params['good_model_id']]

        directory_name = dict_to_result_dir(hyper_params = hyper_params)

        model_file_name =  result_dir_to_file(directory_name = directory_name, output = "params_and_model", 
            uniq_name = uniq_name  , hyper_params = hyper_params)

        # with open( model_file_name, "rb") as f:
        params, model = open_pickled_files(filename = model_file_name)#pickle.load(f)

        if (hyper_params['vi_family'] =="rnvp") & (hyper_params['laplaces_method_use'] == 1): raise ValueError

        metric, metric_std, samples_from_trained_model = metric_iwelbo_eval(params = params, model = model, hyper_params = hyper_params)

        metric_results_file_name =  result_dir_to_file(directory_name = directory_name, output = "metric_results", 
                uniq_name = uniq_name  , hyper_params = hyper_params)
        dump_pickled_files(filename = metric_results_file_name, objects = (metric, metric_std, samples_from_trained_model))
        print ("metric results saved at:", metric_results_file_name)

        metrics_std_list.append(metric_std)
        metrics_list.append(metric)


    if len(metrics_list) == data.shape[0]:

        assert(len(metrics_std_list) == len(metrics_list))
        metrics_list = np.array(metrics_list)
        metrics_std_list = np.array(metrics_std_list)
        # print(metrics_list.shape)
        # print(metrics_std_list.shape)

        assert(metrics_list.shape == (len(data), len(hyper_params['sampling_M_range'])) )

        df_final_metric = pd.concat([ 
                                        pd.DataFrame(data = np.hstack([data.to_numpy(),
                                            np.expand_dims([m]*len(data), 1),
                                            metrics_list[:, i, np.newaxis],
                                            metrics_std_list[:, i, np.newaxis],
                                            ]), 
                                            columns = list(data.columns) + ['sampling_M', 'final_metric', 'final_metric_std'])
                                        for i, m in enumerate(hyper_params['sampling_M_range']) ] 
                                        , ignore_index = True)
        with open("../ezstan/good_model_new_id_to_old_id_dict.pkl", "rb") as f:
            good_model_orig_id_dict = load(f)
            good_model_orig_id_dict_reversed = {i:k for k, i in good_model_orig_id_dict.items()}
        df_final_metric['good_model_id'] = df_final_metric['good_model_id'].apply(lambda x : good_model_orig_id_dict_reversed[x])    

        print(f"Prepared the final metric df with evaluation M in range {hyper_params['sampling_M_range']}")
        print(df_final_metric.head(20))

        # print(df_final_metric.shape)
        save_dataframe(dataframe = df_final_metric, hyper_params = hyper_params, uniq_name = str(uniq_name)+"_final_metric")
        return df_final_metric

    else:
        raise ValueError




def get_final_metrics(hyper_params, uniq_name):
    data = get_training_data(hyper_params.copy(), uniq_name)
    data_cleaned = get_clean_training_data(data)
    final_metric_data = get_final_metric_data(data = data_cleaned, hyper_params = hyper_params.copy(), uniq_name = uniq_name)
    return final_metric_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-uname', '--uname', type=str, required=True) 
    args = parser.parse_args()
    uniq_name = args.uname

    from dict_saver import retrieve_u_name_dict_and_update 
    hyper_params = retrieve_u_name_dict_and_update(uniq_name = uniq_name)
    # sleep(0.5)
    # from config import hyper_params

    final_metric_df = get_final_metrics(hyper_params = hyper_params.copy(), uniq_name = uniq_name)


