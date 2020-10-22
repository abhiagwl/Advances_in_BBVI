import autograd.numpy as np 
# import torch
import math
from pickle import load, dump
import mystan 

def good_iter(i):
    a = 10 ** np.floor(np.log10(i*1.0))
    return (i%a)==0



def evaluate_batch_logp(model,samples):
    B = samples.shape
    assert(B[-1] == model.zlen)
    assert(len(B) == 2)

    cll = []
    for j in range(B[0]):
        cll.append(model.logp(samples[j]))
    cll = np.array(cll) 

    return cll




def running_mean(current_mean, current_x, current_size):
    new_size = current_size+1
    new_mean = (1.0/new_size)*(current_mean[0]*current_size[0] + current_x)
    return [new_mean], [new_size]

def coupling_layer_specifications(hyper_params):
    """
    We specify the FNN based networks over here. A single network produce both s and t parts.
    Coupling Layer currently comprises of 1 full transform but this can be made more complex.
    """
    D = hyper_params['data_dim']
    H = hyper_params['rnvp_num_hidden_units']

    d_1 = np.int(D//2) if D%2 == 0 else np.int(D//2) + 1 
    d_2 = np.int(D - d_1)
    assert(d_1 + d_2 ==  D)
    
    coupling_layer_sizes = []
    coupling_layer_sizes.append([d_1, H, H, 2*d_2])
    coupling_layer_sizes.append([d_2, H, H, 2*d_1])
    return coupling_layer_sizes


def perm_id_to_configuration(perm_id, hyper_params):
    perm  = hyper_params['all_permutations'][perm_id]
    for i, o in enumerate(hyper_params['to_be_permuted_hyper_params_order']):
        hyper_params[o] = perm[i]

    perm_id_to_configuration_further_changes(hyper_params)
def perm_id_to_configuration_further_changes(hyper_params):
    # pass
    if hyper_params['filename'] == 'vi_on_mystan.py':
        with open('./good_model_paths.pkl','rb') as f:
            good_model_paths = load(f)
        del good_model_paths[9]
        model = mystan.get_logp(good_model_paths[hyper_params['good_model_id']])
        hyper_params['data_dim'] = model.zlen
        del model
        hyper_params['step_size'] = hyper_params['step_size']/hyper_params['data_dim']
    elif (hyper_params['filename'] == 'vi_on_mog.py') | (hyper_params['filename'] == 'mle_on_mog.py') :
        raise ValueError
        # hyper_params['data_dim'] = hyper_params['mog_dim']
        # hyper_params['step_size'] = hyper_params['step_size']/hyper_params['data_dim']
    else:
        pass
    # hyper_params['sample_train']
    # if hyper_params['advanced_NFestimators_use']==0:
    hyper_params['sample_size_training'] = hyper_params['sample_memory_budget']//hyper_params['augmentation_param_M']
    assert(hyper_params['sample_memory_budget']>=hyper_params['augmentation_param_M'])
    # else :
    #     raise ValueError
        # hyper_params['sample_size_training'] = hyper_params['sample_memory_budget']//(hyper_params['advanced_NFestimators_stage1_M']*hyper_params['advanced_NFestimators_stage2_M'])
        # assert(hyper_params['sample_memory_budget']>=(hyper_params['advanced_NFestimators_stage1_M']*hyper_params['advanced_NFestimators_stage2_M']))

    # hyper_params['step_size'] = hyper_params['step_size']/hyper_params['data_dim']

def check_order_and_ranges(hyper_params):
    order = hyper_params['to_be_permuted_hyper_params_order']
    print("The order that is to be followed for this run: ", order)
    if not all([o+"_range" in hyper_params.keys() for o in order]) :
        print ("Not all hyper-parameters to be iterated on have the ranges. Taking values from default section and creating listhyper-parameter ranges ")
        for o in order:
            if not (o+"_range" in hyper_params.keys()):
                if o != "seed":
                        hyper_params[o+'_range'] = [hyper_params[o]]
                else :
                        hyper_params[o+'_range'] = np.arange(hyper_params['num_experiment_repeat'])
        all_ranges = [hyper_params[o+"_range"] for o in hyper_params['to_be_permuted_hyper_params_order']]
        hyper_params['all_permutations'] = list(itertools.product(*all_ranges))
        print ("Hyper-parameter ranges have been corrected based on the order")

def print_config(hyper_params):
    for o in hyper_params['to_be_permuted_hyper_params_order']:
        print (o, ": ", hyper_params[o])
    print("Model latent dimensions : ", hyper_params['data_dim'])        
    print("No. of Iterations : ", hyper_params['num_epochs'])
    print("cpu_memory_per_job: ", hyper_params['mem_per_job'])
    print("time allocated for :", hyper_params['time_per_job'])
    print("partition allocated for :", hyper_params['partition_per_job'])
    print ("sample_size_training : ", hyper_params['sample_size_training'])
    to_print = ['laplaces', 'SAA', hyper_params['vi_family']]
    for k in hyper_params.keys():
        if any([tp in k for tp in to_print]):
            print (str(k)+ " : ", hyper_params[k])


def dump_pickled_files(filename, objects, protocol = None):
    with open(filename,"wb") as f:
        if protocol is None:
            dump(objects,f)
        else:
            dump(objects,f, protocol = protocol)

def open_pickled_files(filename, protocol = None):
    with open(filename,"rb") as f:
        if protocol is None:
            objects = load(f)
        else:
            objects = load(f, protocol = protocol)
        return objects

result_dir_hparams = ['good_model_id', 'vi_family', 'grad_estimator_type', 
'laplaces_method_use', 'augmentation_param_M', 'step_size', 'num_epochs', 'seed']

def dict_to_result_dir(hyper_params):
    # if 'to_be_permuted_hyper_params_order' not in hyper_params:
    if 'advi_use' in hyper_params.keys():
        hyper_params['step_size'] = -1
        return ("../data/experiments/"+'/'.join([o+'/'+str(hyper_params[o]) for o in result_dir_hparams])+'/').replace(" ","_")
    return ("../data/experiments/"+'/'.join([o+'/'+str(hyper_params[o]) for o in result_dir_hparams])+'/').replace(" ","_")

    # to_be_permuted_order = hyper_params['to_be_permuted_hyper_params_order']
    # if 'num_epochs_type' in to_be_permuted_order:
    #     to_be_permuted_order.remove('num_epochs_type')
    # return ("../data/experiments/"+'/'.join([o+'/'+str(hyper_params[o]) for o in to_be_permuted_order])+'/').replace(" ","_")


def dict_to_dataframe(hyper_params, results, is_seed):
    # if is_seed ==1 :
    #     r1, r2 = results
    #     return [hyper_params[o] for o in hyper_params['to_be_permuted_hyper_params_order'][:-1]] + list(r1) + list(r2)
    # elif is_seed ==0 :
        # if hyper_params['check_point_use']==1:
        #     return [hyper_params[o] for o in hyper_params['to_be_permuted_hyper_params_order']] + list() + list(results)
        # else :
    r1,r2 = results 
    assert(r1 == r2)
    return [hyper_params[o] for o in result_dir_hparams] + [r1]


def result_dir_to_file(directory_name, output, uniq_name, hyper_params):
    #TODO check the name of the file
    # if hyper_params['mog_use'] ==1 : 
    #     if uniq_name is not None:
    #         file_name = directory_name + output+'_'+"mog_"+str(hyper_params['mog_K'])+"_"+uniq_name+'.pkl'
    #     else:
    #         file_name = directory_name + output+'_'+"mog_"+str(hyper_params['mog_K'])+'.pkl'
    # else : 
    if uniq_name is not None:
        file_name = directory_name + output+'_'+str(hyper_params['good_model_id'])+"_"+uniq_name+'.pkl'
    else:
        file_name = directory_name + output+'_'+str(hyper_params['good_model_id'])+'.pkl'
    return file_name 

def dict_to_arguments(argument_dict):
    return " "+' '.join([str(k)+' '+str(argument_dict[k]) for k in argument_dict.keys()])

def generate_run_command(environment, file, argument_dict):
    return 'source activate '+environment+'\npython '+file+ dict_to_arguments(argument_dict) +'\nexit'
