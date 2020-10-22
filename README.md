# Advances in Black-Box VI: Normalizing Flows, Importance Weighting, and Optimization

This is the supplementary code for the NeurIPS submission 3269

## Requirements

To create a conda environment similar to our set-up use the ```env.yml``` file provided with in ```ezstan``` directory (need to change directory to ezstan to run the command as is):


```setup
conda env create --file env.yml
```

> 📋 The set-up requires:
> - Autograd for automatic differentiation
> - Pystan and rpy2 to process Stan models 
> - Pandas, Plotnine, matplotlib, Numpy for analyzing results.  

After creating the conda environment, activate the environment by following the command displayed (this will be either ```conda activate adv_in_bbvi_env``` or ```source activate adv_in_bbvi_env```). Our set-up consists of two directories, `ezstan` and `data`. `ezstan` contains all the necessary code for replicating our results on the Stan models from the paper. `data` acts as the collection directory for parameters, final result dataframes, and figures.


## Train inference methods
### Run inference schemes from the paper
To run our inference scheme, execute the following command from the ```ezstan``` directory:

```train
python vi_on_mystan.py --models 1 2 3 --uname first_vi_run
```

> Will run inference on models with Id 1, 2, and 3 in sequence. You can change the default hyper-parameter setting in  ```config.py```. ```models``` takes the Id number of models in the benchmark. Please refer to table 1 in the Appendix for full details of the models. ```uname``` is used to characterize the run for easier retrieval of results. For all other configurations, please take a look at the detailed comments in the ```config.py```. Training results, model, and parameters are saved in `data/experiments/` directory based on the configurations of the hyper-parameters. 




### Run ADVI
To run our implementation of ADVI, run this command from the ```ezstan``` directory:

```train
python advi_on_mystan.py --models 1 2 3 --uname first_advi_run
```

> Will run ADVI models with Id 1, 2, and 3 in sequence. You can change the default ADVI specific hyper-parameter setting in  ```advi_on_mystan.py```. 


## Evaluate metrics

You can use the following script to evaluate the performance metric corresponding to a particular inference run: 

```eval
python get_metrics.py --uname first_vi_run
```
> This save a pandas dataframe at ```data/experiments/dataframe/``` for the run with uname = 'first_vi_run'.


## Replicating complete results
The above scripts are sufficient to produce all the results we use in the paper; however, we do not include a script that runs over all relevant permutations and re-creates the complete set of results. This is deliberate as the total number of jobs is in excess of 1,000. We believe the optimal way to re-run the complete experiment would be to use a server-cluster and concatenate the final metric dataframes returned from the ```get_metrics.py``` for different methods and configurations.


However, we do include the final metric dataframe that was used to generate the tables and figures from the paper; this dataframe is available at ```/data/df_all_results_final_metrics.csv```. We also include the scripts to re-create the plots from this final dataframe; use the following command to recreate all plots:


```plots
python myplots.py -df_fname ../data/df_all_results_final_metrics.csv -path 1 -ablation 1 -pair 1
```
> This command takes the final_metric dataframe and creates the final figures for the path study, ablation study (both in-text figures and extended figures), and pairwise comparison figures. All the figures will be saved as `.pdf` files in `data/experiments/figures`

