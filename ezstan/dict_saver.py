from utilities.result_helper import save_dict, dict_dir_name, dict_file_name
import argparse
from pickle import load, dump
import os

def save_current_dict(hyper_params, uniq_name):
    print("Saving the hyper_params dictionary for easy retrieval")
    from utilities.helper import dump_pickled_files 
    # dump_pickled_files(objects = hyper_params, filename = "./hyper_param_test.pkl")
    dict_dirname = dict_dir_name()
    if not os.path.exists(dict_dirname):
        os.makedirs(dict_dirname)

    dump_pickled_files(filename = dict_file_name(dir_name = dict_dirname, uniq_name = uniq_name),
        objects = hyper_params)
    print("dict saved at : ", dict_file_name(dir_name = dict_dirname, uniq_name = uniq_name))


def retrieve_u_name_dict_and_update(uniq_name):
    from utilities.helper import open_pickled_files 

    dict_dirname = dict_dir_name()
    if not os.path.exists(dict_dirname):
        os.makedirs(dict_dirname)

    return open_pickled_files(filename = dict_file_name(dir_name = dict_dirname, uniq_name = uniq_name))
