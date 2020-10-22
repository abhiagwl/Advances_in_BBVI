import os
import pickle
import autograd.numpy as np 
from vi_families import gaussian_sample, gaussian_logpdf, rnvp_sample_logpdf, generate_mask, mix_of_gaussian_logpdf
from utilities.helper import evaluate_batch_logp, dict_to_result_dir, result_dir_to_file, dump_pickled_files
from autograd.scipy.special import logsumexp
from autograd.core import getval

def results_parameters(hyper_params, model, params):
    if hyper_params['memory_efficient_evaluation'] == 0: 
        if hyper_params['vi_family'] == "gaussian" :
            samples = gaussian_sample(mu = params[0], sig = np.tril(params[1]), 
                sample_size = (hyper_params['sample_size_evaluation'],hyper_params['augmentation_param_M']))
            lq = gaussian_logpdf(samples = samples, mu = params[0], sig = np.tril(params[1]))
        elif hyper_params['vi_family'] == "rnvp" :
            m = generate_mask(hyper_params = hyper_params)
            if hyper_params['SAA_use'] == 1:
                hyper_params['SAA_use'] = 0
                samples, lq = rnvp_sample_logpdf(params = params, mask = m, hyper_params = hyper_params,
                    sample_size = (hyper_params['sample_size_evaluation'],hyper_params['augmentation_param_M']))
                hyper_params['SAA_use'] = 1
            else:
                samples, lq = rnvp_sample_logpdf(params = params, mask = m, hyper_params = hyper_params,
                    sample_size = (hyper_params['sample_size_evaluation'],hyper_params['augmentation_param_M']))



        #stan currently doesnot support batching of inputs for logpdf calculation
        #iterative scheme to calculate for a batch
        if hyper_params['mog_use']==1:
            cll = mix_of_gaussian_logpdf(samples = samples, normals = hyper_params['mog_normals'], a = hyper_params['mog_a'])
        else:

            cll = []
            for i in range(hyper_params['sample_size_evaluation']):
                cll.append(evaluate_batch_logp(model, samples[i]))
            cll = np.array(cll)



        assert(cll.shape == lq.shape)
        assert(cll.shape == (hyper_params['sample_size_evaluation'],hyper_params['augmentation_param_M']))
        
        iw_elbo = -1.0*np.log(hyper_params['augmentation_param_M']) + logsumexp(cll - lq, -1)
        
        assert(iw_elbo.shape == (hyper_params['sample_size_evaluation'],))
        if hyper_params['mog_use'] == 1:
            return_item =  [-1.0*np.nanmean(iw_elbo)]
        else :
            return_item =  [np.nanmean(iw_elbo)]
    elif hyper_params['memory_efficient_evaluation'] == 1 : 
        # return_item = [-1.0*ELBO.item()]
        # we want to do this in the same memory budget as the training part
        return_item = []
        for i in range(hyper_params['sample_size_evaluation']//hyper_params['sample_size_training']):

            if hyper_params['vi_family'] == "gaussian" :
                samples = gaussian_sample(mu = params[0], sig = np.tril(params[1]), 
                    sample_size = (hyper_params['sample_size_training'],hyper_params['augmentation_param_M']))
                lq = gaussian_logpdf(samples = samples, mu = params[0], sig = np.tril(params[1]))
            elif hyper_params['vi_family'] == "rnvp" :
                m = generate_mask(hyper_params = hyper_params)
                if hyper_params['SAA_use'] == 1:
                    hyper_params['SAA_use'] = 0
                    samples, lq = rnvp_sample_logpdf(params = params, mask = m, hyper_params = hyper_params,
                        sample_size = (hyper_params['sample_size_training'],hyper_params['augmentation_param_M']))
                    hyper_params['SAA_use'] = 1
                else:
                    samples, lq = rnvp_sample_logpdf(params = params, mask = m, hyper_params = hyper_params,
                        sample_size = (hyper_params['sample_size_training'],hyper_params['augmentation_param_M']))
        
            if hyper_params['mog_use']==1:
                cll = mix_of_gaussian_logpdf(samples = samples, normals = hyper_params['mog_normals'], a = hyper_params['mog_a'])
            else:

                cll = []
                for i in range(samples.shape[0]):
                    cll.append(evaluate_batch_logp(model, samples[i]))
                cll = np.array(cll)



            assert(cll.shape == lq.shape)
            assert(cll.shape == (hyper_params['sample_size_training'],hyper_params['augmentation_param_M']))
            
            iw_elbo = -1.0*np.log(hyper_params['augmentation_param_M']) + logsumexp(cll - lq, -1)
            
            assert(iw_elbo.shape == (hyper_params['sample_size_training'],))
            if hyper_params['mog_use'] == 1:
                return_item.append(-1.0*iw_elbo)
            else :
                return_item.append(iw_elbo)
        return_item = np.ravel(return_item)
        assert((len(return_item) == hyper_params['sample_size_evaluation']) | (len(return_item) == (hyper_params['sample_size_evaluation']//hyper_params['sample_size_training'])*hyper_params['sample_size_training']))
        return_item = [np.nanmean(return_item)]

    return (return_item)

def error_results_parameters(hyper_params, model, params):
    return ([np.nan])


def save_results_parameters(hyper_params, model, params, uniq_name, results, time):
    # if results is None:
    params = getval(params)
    train_results = results
 
    # final_model_results = results_parameters(hyper_params, model, params)
    # else:
    # train_average_results = [np.nanmean(train_results,0)]

    # results = [final_model_results[0], train_average_results[0]]
    results = [np.nanmean(train_results,0), np.nanmean(train_results,0)]
    
    directory_name = dict_to_result_dir(hyper_params)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    result_file_name =  result_dir_to_file(directory_name = directory_name, output = "results", uniq_name = uniq_name, hyper_params = hyper_params)
    print("average training result, average training result: ", results)
    dump_pickled_files(filename = result_file_name, objects = (results, time))
    print ("results saved at:", result_file_name)

    # print("last 10 results: ", results[-10:])


    train_results_file_name =  result_dir_to_file(directory_name = directory_name, output = "train_results", uniq_name = uniq_name, hyper_params = hyper_params)
    dump_pickled_files(filename = train_results_file_name, objects = train_results)
    print ("train_results saved at:", train_results_file_name)
    
    # results_ = pickle.load(open(result_file_name,"rb"))
    # print (results_)
    # f = open(result_file_name, "wb")
    # pickle.dump(results, f)
    # f.close()
    # hyper_param_dir_name = uniq_name_to_hyper_dir(uniq_name = uniq_name)
    # if not os.path.exists(hyper_param_dir_name):
    #     os.makedirs(hyper_param_dir_name)
    # hyper_param_file_name =  hyper_dir_to_file_name(dir_name = hyper_param_dir_name, uniq_name = uniq_name )
    # pickle.dump(hyper_params, open(hyper_param_file_name, "wb"))
    # dump_pickled_files(filename = hyper_param_file_name, objects = hyper_params,  protocol=pickle.HIGHEST_PROTOCOL)
    # print ("Hyper-paramters saved at:", hyper_param_file_name)
    
    model_file_name =  result_dir_to_file(directory_name = directory_name, output = "params_and_model", uniq_name = uniq_name, hyper_params = hyper_params)
    dump_pickled_files(filename = model_file_name, objects = (params,model))
    print ("model saved at:", model_file_name)

    del results

def save_error_results_parameters(hyper_params, model, params, uniq_name):

    results = error_results_parameters(data, hyper_params, model, q)
    directory_name = dict_to_result_dir(hyper_params)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    result_file_name =  result_dir_to_file(directory_name = directory_name, output = "results", uniq_name = uniq_name, hyper_params = hyper_params)
    print(results)
    pickle.dump(results, open(result_file_name, "wb"))
    print ("results saved at:", result_file_name)
    hyper_param_file_name =  result_dir_to_file(directory_name = directory_name, output = "hyper_params", uniq_name = uniq_name, hyper_params = hyper_params)
    pickle.dump(hyper_params, open(hyper_param_file_name, "wb"))
    print ("paramters saved at:", hyper_param_file_name)
    del results

def dataplot_dir_name(uniq_name):
    plot_directory_name = '../data/experiments/plots/'+uniq_name+"/"
    return plot_directory_name

def dataplot_file_name(dir_name, uniq_name):
    file_name = dir_name +  uniq_name +"_plot.png"
    return file_name

def uniq_name_to_hyper_dir(uniq_name):
    hyper_params_directory_name = '../data/experiments/hyper_params/'+uniq_name+"/"
    return hyper_params_directory_name

def hyper_dir_to_file_name(dir_name, uniq_name):
    file_name = dir_name +  uniq_name +"_hyper_params.pkl"
    return file_name


def dict_dir_name():
    dict_directory_name = '../data/experiments/dictionaries/'
    return dict_directory_name

def dict_file_name(dir_name, uniq_name):
    # frame_name = [o +"_"+ str((hyper_params[o+"_range"][0], hyper_params[o+"_range"][-1])) + "_"  for o in hyper_params['to_be_permuted_hyper_params_order']]
    file_name = dir_name +  uniq_name +"_dict.pkl"
    return file_name

def save_dict(dict_text, uniq_name):
    dict_dirname = dict_dir_name()
    if not os.path.exists(dict_dirname):
        os.makedirs(dict_dirname)
    dump_pickled_files(filename = dict_file_name(dir_name = dict_dirname, uniq_name = uniq_name),
        objects = dict_text)
    print("dict saved at : ", dict_file_name(dir_name = dict_dirname, uniq_name = uniq_name))

def dataframe_dir_name():
    data_frame_directory_name = '../data/experiments/dataframe/'
    return data_frame_directory_name

def dataframe_file_name(dir_name, uniq_name):
    # frame_name = [o +"_"+ str((hyper_params[o+"_range"][0], hyper_params[o+"_range"][-1])) + "_"  for o in hyper_params['to_be_permuted_hyper_params_order']]
    file_name = dir_name +  uniq_name +"_data_frame.pkl"
    return file_name

def save_dataframe(dataframe, hyper_params, uniq_name):
    dataframe_dirname = dataframe_dir_name()
    if not os.path.exists(dataframe_dirname):
        os.makedirs(dataframe_dirname)
    dump_pickled_files(filename = dataframe_file_name(dir_name = dataframe_dirname, uniq_name = uniq_name),
        objects = dataframe)
    print("dataframe saved at : ",  dataframe_file_name(dir_name = dataframe_dirname, uniq_name = uniq_name))
    # pickle.dump(dataframe,open(dataframe_file_name() , 'wb'))
    # store_laplaces_init, retrieve_stored_laplaces_init

def laplaces_init_dir(hyper_params):
    laplaces_init_dir_name = '../data/experiments/models/laplaces_init/' + str(hyper_params['good_model_id'])
    return laplaces_init_dir_name

def laplaces_init_file(dir_name):
    laplaces_init_file_name = dir_name + "params.pkl"
    return laplaces_init_file_name

def check_laplaces_init_stored(hyper_params):
    laplaces_init_dirname = laplaces_init_dir(hyper_params = hyper_params)
    file_name = laplaces_init_file(dir_name = laplaces_init_dirname)
    return 1 if os.path.exists(file_name) else 0

def retrieve_stored_laplaces_init(hyper_params):
    laplaces_init_dirname = laplaces_init_dir(hyper_params = hyper_params)
    file_name = laplaces_init_file(dir_name = laplaces_init_dirname)
    if not os.path.exists(file_name):
        print("Trying to retrieve before saving laplaces init params")
        exit()
    print("Loading the saved LAI initialization values.")
    with open(file_name, "rb") as f:
        params = pickle.load(f)
    return params

def store_laplaces_init(params,  hyper_params):  
    laplaces_init_dirname = laplaces_init_dir(hyper_params = hyper_params)
    if not os.path.exists(laplaces_init_dirname):
        os.makedirs(laplaces_init_dirname)
    dump_pickled_files(filename = laplaces_init_file(dir_name = laplaces_init_dirname),
        objects = params)