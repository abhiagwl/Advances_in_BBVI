import pandas as pd
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from plotnine import *

from matplotlib import rc
rc('text', usetex=True)
import matplotlib as mpl
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathtools}']

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-df_fname', '--df_fname', type=str, required=False, default = "../data/df_all_results_final_metrics.csv") 
parser.add_argument('-path', '--path', type=int, required=False, default = 0) 
parser.add_argument('-ablation', '--ablation', type=int, required=False, default = 0) 
parser.add_argument('-pair', '--pair', type=int, required=False, default = 0) 
args = parser.parse_args()
df_fname = args.df_fname
path_plots = args.path
ablation_plots = args.ablation
ablation_plots = args.ablation
pair_plots = args.pair

from utilities.helper import open_pickled_files

def get_single_table(df, metric) :
    model_list = [2, 30, 80, 118, 17, 5, 65, 26, 44, 32, 78, 29, 128, 7, 84, 28, 41, 
                  136, 3, 131, 155, 103, 10, 53, 91, 89, 125, 124, 112, 151]

    g_names = np.array(open_pickled_files("../ezstan/good_model_name.pkl"))
    g_lens = np.array(open_pickled_files("../ezstan/good_model_length.pkl"))
    g_data_lens = (open_pickled_files("../ezstan/good_model_data_length.pkl"))


    data_len = np.array([g_data_lens[m] for m in model_list])
    z_len = g_lens[model_list]
    ratio = z_len/data_len
    model_list = np.array(model_list)
    model_list = model_list[list(reversed(np.argsort(ratio)))]
    model_num_dict = {m:i+1 for i, m in enumerate(model_list)}
    df['good_model_id'] = df['good_model_id'].apply(lambda x: model_num_dict[x] if x in model_num_dict else x)

    vfam_dict = {'gaussian': "Full-rank Gaussian", "rnvp": "Real NVP flows"}
    df['vi_family'] = df['vi_family'].apply(lambda x : vfam_dict[x] if x in vfam_dict else x)
    #     df[df['laplaces_method_use'] == 0]
    col_dict = {
        'good_model_names': 'Model Name', 
        'good_model_id': 'Id', 
        'augmentation_param_M': "IWVI M$_{\\text{training}}$",
        'vi_family':'$q_{\phi}$ family', 
        'grad_estimator_type':'${\\nabla}_{\phi}$', 
        'laplaces_method_use':'LI', 
        'seed': "Random seed",
        'sampling_M': "IWVI M$_{\\text{sampling}}$",
        'step_size_scheme': "Step-size scheme",
         }
    df = df.rename(columns = col_dict)
    table = pd.pivot_table(df, values=metric, 
                           columns=[ 
                                    col_dict['vi_family'], 
                                    col_dict['grad_estimator_type'], 
                                    col_dict['laplaces_method_use'],
                                    col_dict['augmentation_param_M'],
                                    col_dict['sampling_M'],
                                    col_dict['seed'], 
                                    col_dict['step_size_scheme'], 
                           ],
                           index=[col_dict['good_model_id'], col_dict['good_model_names']],
                           aggfunc= np.nanmax)
    return table
# if ablation_plots:
df_final_metric = pd.read_csv(df_fname)
table = get_single_table(df_final_metric.copy(), 'final_metric')
# print(table.head())
# exit()
if not os.path.exists("../data/experiments/figures"): os.makedirs("../data/experiments/figures")
################################################################################
########        Generate Path Figure from final metric    ######################
################################################################################



def get_path_Δ_df(table, M_sample_baseline, M_sample_path_IWVI, M_sample_path_VI, seed, max_Δ = 100.0):
    Δ = pd.DataFrame(columns = ['tag', 'Δ'])
    
    baseline = ('Full-rank Gaussian', 'closed-form-entropy', 0, 1, M_sample_baseline, seed, "ADVI")
    path_stop_4c = ("Method (4c)", 'Real NVP flows', "IWAEDREG", 0, 1, M_sample_path_IWVI, seed, "comprehensive step search")
    path_stop_4b = ("Method (4b)", 'Real NVP flows', "IWAEDREG", 0, 1, M_sample_path_VI, seed, "comprehensive step search")
    path_stop_4a = ("Method (4a)", 'Real NVP flows', "IWAE", 0, 1, M_sample_path_VI, seed, "comprehensive step search")
    path_stop_3a = ("Method (3a)",'Full-rank Gaussian', "IWAEDREG", 0, 1, M_sample_path_IWVI, seed, "comprehensive step search")
    path_stop_1 = ("Method (1)", 'Full-rank Gaussian', "IWAEDREG", 0, 1, M_sample_path_VI, seed, "comprehensive step search")
    path_stop_0 = ("Method (0)", 'Full-rank Gaussian', "closed-form-entropy", 0, 1, M_sample_path_VI, seed, "comprehensive step search")

    path_stops = [path_stop_0, path_stop_1, path_stop_3a, path_stop_4a, path_stop_4b, path_stop_4c]

    def get_path_stop_Δ(Δ, path_stop, baseline):
        Δ = Δ.append(pd.DataFrame(
                    data = np.array([
                                    [path_stop[0]]*30 ,
                                    (table[path_stop[1:]] - 
                                     table[baseline]).fillna(0).apply(lambda x: np.minimum(x, max_Δ))
                                    ]).T,
                    columns = Δ.columns
                    ), ignore_index= True)
        return Δ
    for p_stop in path_stops:
        Δ = get_path_stop_Δ(Δ.copy(), p_stop, baseline)

    Δ['Δ'] = Δ['Δ'].astype('float')
    return Δ


def plot_Δ_cdf(Δ, vis_max_Δ, seed):
    return (ggplot(Δ)
 +stat_ecdf(mapping=aes(x = 'Δ', color = 'tag', size = 'tag', alpha = 'tag'), geom='step', position='identity',na_rm=False)
 +scale_y_continuous(breaks = np.linspace(0, 1, 6), minor_breaks = 5)
 +scale_x_continuous(breaks = np.linspace(0, vis_max_Δ, 6), minor_breaks = 3)

 +coord_cartesian(xlim=[-0.5, vis_max_Δ], expand = False)
 +theme(panel_grid=element_line(color="lightgray"),
        figure_size=(12, 6), 
        legend_position=(0.6, 0.3),
        legend_box=element_blank(),
        legend_title=element_blank(),
        plot_title=element_blank(),
       )
 +guides(
     color=guide_legend(nrow=2, keywidth = 20, keyheight = 4), 
     size=guide_legend(nrow=2, keywidth = 20, keyheight = 4 ), 
     alpha=guide_legend(nrow=2, keywidth = 20, keyheight = 4 ) 
 )
 
 +labs(x="$\Delta$ (in nats)\n $\Delta \coloneqq$ Lower bound improvement over Gaussian VI w/o LI", 
       y="Frac. models with $\Delta$ improvement or more",
      )
 
 +scale_size_manual(np.array([2]*7)*0.75)
 +scale_color_manual(["#E69F00", "#56B4E9", "#009E73","#D55E00", "#BBB019", "#CC79A7", "#0072B2"])
 +scale_alpha_manual([0.75]*7)
)

def plot_path_Δ_ccdf(Δ, seed ,vis_min_Δ = -0.25, vis_max_Δ = 5, vis_step = 1):
    labels = sorted(Δ['tag'].unique())
    p = plot_Δ_cdf(Δ.copy(), vis_max_Δ=vis_max_Δ, seed =seed)
    p._build()
    dfp = p.layers[0].data
    dfp['x'] = pd.to_numeric(dfp['x'])
    dfp['y'] = pd.to_numeric(dfp['y'])
    dfp['y'] = 1 - dfp['y']
    
    for g in dfp['group'].unique():
        d = dfp[dfp['group'] == g].iloc[0, :]
        d['x'] = -1000.0
        d['y'] = 1.0
        dfp = dfp.append(d, ignore_index = True)
        d['x'] = 1000.0
        d['y'] = 0.0
        dfp = dfp.append(d, ignore_index = True)

    dfp['group'] = dfp['group'].astype('str')
    dfp['x'] = pd.to_numeric(dfp['x'])
    dfp['y'] = pd.to_numeric(dfp['y'])
    p_ = (ggplot(dfp)
        +scale_y_continuous(breaks = np.linspace(0, 1, 6), 
                            minor_breaks = 5)
        +scale_x_continuous(breaks = np.arange(vis_min_Δ, vis_max_Δ+1, vis_step) if vis_min_Δ<=-1 else np.arange(0, vis_max_Δ+1, vis_step), 
                            minor_breaks = 4)
        +geom_step(mapping = aes(x = 'x', y = 'y', 
                                 color = 'group', alpha = 'group', 
                                 size = 'group')) 
        +coord_cartesian(xlim=[vis_min_Δ, vis_max_Δ], ylim = [0,1], expand = False)
         +theme(
                panel_border = element_rect(color = 'black'),
                panel_grid=element_line(color = 'lightgray'),
                figure_size=(6.5, (21/34)*6.5), 
                legend_position=(0.65, 0.8),
                legend_background=element_rect(color = 'black'),
                legend_title=element_blank(),
                legend_box_margin = 2,
                plot_title=element_blank(),
                plot_margin=0,
                legend_text=element_text(size = 15),
                axis_text=element_text(size = 15),
                axis_title=element_text(size = 18),
                axis_ticks_minor_x=element_line(color = "black"),
                axis_ticks_minor_y=element_line(color = "black"),
               )
         +guides(
             color=guide_legend(nrow=2, keywidth = 20, keyheight = 4), 
             size=guide_legend(nrow=2, keywidth = 20, keyheight = 4 ), 
             alpha=guide_legend(nrow=2, keywidth = 20, keyheight = 4 ) 
         )
          +geom_segment(aes(x=-0.34, y = 0.06, xend = 0.11, yend = 0.246), arrow = arrow(length= 0.1, type = 'closed'))
          +annotate('text', x = 0.81, y = 0.09, label = "(Direction of better performance)", size = 15)
          
          

         +labs(x="$\Delta \coloneqq$ Improvement over ADVI (nats)", 
               y="Frac. models with $\Delta$ improvement or more",
              )
    
         +scale_size_manual(np.array([2.5]*7), labels = labels)
         +scale_color_manual(["#92b73e", "#E69F00", "#D55E00", "#56B4E9", "#009E73", "#0072B2","#CC79A7", "#BBB019"], labels = labels)
          
         +scale_alpha_manual([0.65]*7, labels = labels)
     )
    return p_
if path_plots: 
	save_as_pdf_pages([
	        plot_path_Δ_ccdf(get_path_Δ_df(table.copy(), M_sample_baseline=1, M_sample_path_IWVI=10, M_sample_path_VI=1, seed=s), 
	                vis_max_Δ=v, seed=s, vis_min_Δ= -0.5, vis_step=1)
				    for s in [77, 12, 11] for v in [4]
				], filename="path_figure_ccdf.pdf", path="../data/experiments/figures")

# exit()
################################################################################
########        Generate Ablation Figure 10 from final metric    ###############
################################################################################
def plot_ablation_intext_Δ_ccdf(Δ, vis_max_Δ, seed):
    labels = sorted(Δ['tag'].unique())
    p = plot_Δ_cdf(Δ.copy(), vis_max_Δ=vis_max_Δ, seed =seed)
    p._build()
    dfp = p.layers[0].data
    dfp['x'] = pd.to_numeric(dfp['x'])
    dfp['y'] = pd.to_numeric(dfp['y'])
    dfp['y'] = 1 - dfp['y']
    
    for g in dfp['group'].unique():
        d = dfp[dfp['group'] == g].iloc[0, :]
        d['x'] = -1000.0
        d['y'] = 1.0
        dfp = dfp.append(d, ignore_index = True)

    dfp['group'] = dfp['group'].astype('str')
    dfp['x'] = pd.to_numeric(dfp['x'])
    dfp['y'] = pd.to_numeric(dfp['y'])
    p_ = (ggplot(dfp)
        +scale_y_continuous(breaks = np.linspace(0, 1, 6), 
                            minor_breaks = 5)
        +scale_x_continuous(breaks = np.linspace(0, vis_max_Δ, 6), 
                            minor_breaks = 4)
        +geom_step(mapping = aes(x = 'x', y = 'y', 
                                 color = 'group', alpha = 'group', 
                                 size = 'group')) 
        +coord_cartesian(xlim=[-0.25, vis_max_Δ], ylim = [0,1], expand = False)
         +theme(
                panel_border = element_rect(color = 'black'),
                panel_grid = element_line(color="lightgray"),
                figure_size=(4.4, 4.0), 
                legend_position=(0.58, 0.9),
                legend_background=element_rect(color = 'black'),
                legend_title=element_blank(),
                legend_box_margin = 1,
                plot_title=element_blank(),
                plot_margin=0,
                legend_text=element_text(size = 13),
                axis_text=element_text(size = 15),
                axis_title=element_text(size = 16),
                axis_ticks_minor_x=element_line(color = "black"),
                axis_ticks_minor_y=element_line(color = "black"),
               )
         +guides(
             color=guide_legend(nrow=2, keywidth = 20, keyheight = 4), 
             size=guide_legend(nrow=2, keywidth = 20, keyheight = 4 ), 
             alpha=guide_legend(nrow=2, keywidth = 20, keyheight = 4 ) 
         )

         +labs(x="$\Delta \coloneqq$ Improvement over Method (1)", 
               y="Frac. models with $\Delta$ improvement or more",
              )
    
         +scale_size_manual(np.array([2.5]*7), labels = labels)
         +scale_color_manual([ "#D55E00", "#009E73","#56B4E9",  "#0072B2","#BBB019", "#E69F00", "#CC79A7"], labels = labels)
         +scale_alpha_manual([0.65]*7, labels = labels)
     )
    return p_

def get_ablation_intext_Δ_df(table, M_sample_baseline, M_sample_path_IWVI, M_sample_path_VI, seed, max_Δ = 100.0):
    Δ = pd.DataFrame(columns = ['tag', 'Δ'])
    
    baseline = ('Full-rank Gaussian', "IWAEDREG", 0, 1, M_sample_baseline, seed, "comprehensive step search")
    best = ("Best model--(4c)", 'Real NVP flows', "IWAEDREG", 0, 1, M_sample_path_IWVI, seed, "comprehensive step search")
    ablate_IW_sample = ("Ablate IW sampling", 'Real NVP flows', "IWAEDREG", 0, 1, M_sample_path_VI, seed, "comprehensive step search")
    ablate_STL = ("Ablate STL", 'Real NVP flows', "IWAE", 0, 1, M_sample_path_IWVI, seed, "comprehensive step search")
    ablate_Flows = ("Ablate Flows", 'Full-rank Gaussian', "IWAEDREG", 0, 1, M_sample_path_IWVI, seed, "comprehensive step search")
    
    def get_path_stop_Δ(Δ, path_stop, baseline):
        Δ = Δ.append(pd.DataFrame(
                    data = np.array([
                                    [path_stop[0]]*30 ,
                                    (table[path_stop[1:]] - 
                                     table[baseline]).fillna(0).apply(lambda x: np.minimum(x, max_Δ))
                                    ]).T,
                    columns = Δ.columns
                    ), ignore_index= True)
        return Δ
    Δ = get_path_stop_Δ(Δ.copy(), ablate_IW_sample, baseline)
    Δ = get_path_stop_Δ(Δ.copy(), ablate_Flows, baseline)
    Δ = get_path_stop_Δ(Δ.copy(), ablate_STL, baseline)
    Δ = get_path_stop_Δ(Δ.copy(), best, baseline)

    Δ['Δ'] = Δ['Δ'].astype('float')
    return Δ

if ablation_plots:
	save_as_pdf_pages([
	        plot_ablation_intext_Δ_ccdf(get_ablation_intext_Δ_df(table.copy(), M_sample_baseline=1, M_sample_path_IWVI=10, M_sample_path_VI=1, seed=s), 
	                vis_max_Δ=v, seed=s)
	    for s in [77, 12, 11] for v in [5] 
	], filename="ablation_intext_ccdf.pdf", path="../data/experiments/figures/")

# exit()
################################################################################
########        Generate Extended Ablation Figure from final metric    #########
################################################################################
def plot_ablation_extended_Δ_ccdf(Δ, vis_max_Δ, seed):
    labels = sorted(Δ['tag'].unique())
    p = plot_Δ_cdf(Δ.copy(), vis_max_Δ=vis_max_Δ, seed =seed)
    p._build()
    dfp = p.layers[0].data
    dfp['x'] = pd.to_numeric(dfp['x'])
    dfp['y'] = pd.to_numeric(dfp['y'])
    dfp['y'] = 1 - dfp['y']
    
    for g in dfp['group'].unique():
        d = dfp[dfp['group'] == g].iloc[0, :]
        d['x'] = -1000.0
        d['y'] = 1.0
        dfp = dfp.append(d, ignore_index = True)

    dfp['group'] = dfp['group'].astype('str')
    dfp['x'] = pd.to_numeric(dfp['x'])
    dfp['y'] = pd.to_numeric(dfp['y'])
    p_ = (ggplot(dfp)
        +scale_y_continuous(breaks = np.linspace(0, 1, 6), 
                            minor_breaks = 5)
        +scale_x_continuous(breaks = np.arange(0, vis_max_Δ + 1, 1), 
                            minor_breaks = 1)
        +geom_step(mapping = aes(x = 'x', y = 'y', 
                                 color = 'group', alpha = 'group', 
                                 size = 'group')) 
        +coord_cartesian(xlim=[-0.5, vis_max_Δ], ylim = [0,1], expand = False)
         +theme(
                panel_border = element_rect(color = 'black'),
                panel_grid = element_line(color="lightgray"),
                figure_size=(6.5, (21/34)*6.5), 
                legend_position=(0.55, 0.88),
                legend_background=element_rect(color = 'black'),
                legend_title=element_blank(),
                legend_box_margin = 1,
                plot_title=element_blank(),
                plot_margin=0,
                legend_text=element_text(size = 14),
                axis_text=element_text(size = 15),
                axis_title=element_text(size = 17),
                axis_ticks_minor_x=element_line(color = "black"),
                axis_ticks_minor_y=element_line(color = "black"),
               )
         +guides(
             color=guide_legend(nrow=2, keywidth = 20, keyheight = 4), 
             size=guide_legend(nrow=2, keywidth = 20, keyheight = 4 ), 
             alpha=guide_legend(nrow=2, keywidth = 20, keyheight = 4 ) 
         )
         +geom_segment(aes(x=-0.31, y = 0.08, xend = 0.14, yend = 0.266), arrow = arrow(length= 0.1, type = 'closed'))
         +annotate('text', x = 0.85, y = 0.11, label = "(Direction of better performance)", size = 15)
          

         +labs(x="$\Delta \coloneqq$ Improvement over Method (1)", 
               y="Frac. models with $\Delta$ improvement or more",
              )
    
         +scale_size_manual(np.array([2.5]*7), labels = labels)
         +scale_color_manual([ "#D55E00", "#009E73","#56B4E9",  "#0072B2","#BBB019", "#E69F00", "#CC79A7"], labels = labels)
         +scale_alpha_manual([0.65]*7, labels = labels)
     )
    return p_

def get_ablation_extended_Δ_df(table, M_sample_baseline, M_sample_path_IWVI, M_sample_path_VI, seed, max_Δ = 100.0):
    Δ = pd.DataFrame(columns = ['tag', 'Δ'])
    
    baseline = ('Full-rank Gaussian', "IWAEDREG", 0, 1, M_sample_baseline, seed, "comprehensive step search")
    best = ("Best model--(4c)", 'Real NVP flows', "IWAEDREG", 0, 1, M_sample_path_IWVI, seed, "comprehensive step search")
    ablate_IW_sample = ("Ablate IW sampling", 'Real NVP flows', "IWAEDREG", 0, 1, M_sample_path_VI, seed, "comprehensive step search")
    ablate_STL = ("Ablate STL", 'Real NVP flows', "IWAE", 0, 1, M_sample_path_IWVI, seed, "comprehensive step search")
    ablate_Flows = ("Ablate Flows", 'Full-rank Gaussian', "IWAEDREG", 0, 1, M_sample_path_IWVI, seed, "comprehensive step search")
    
    def get_path_stop_Δ(Δ, path_stop, baseline):
        Δ = Δ.append(pd.DataFrame(
                    data = np.array([
                                    [path_stop[0]]*30 ,
                                    (table[path_stop[1:]] - 
                                     table[baseline]).fillna(0).apply(lambda x: np.minimum(x, max_Δ))
                                    ]).T,
                    columns = Δ.columns
                    ), ignore_index= True)
        return Δ
    Δ = get_path_stop_Δ(Δ.copy(), ablate_IW_sample, baseline)
    Δ = get_path_stop_Δ(Δ.copy(), ablate_Flows, baseline)
    Δ = get_path_stop_Δ(Δ.copy(), ablate_STL, baseline)
    Δ = get_path_stop_Δ(Δ.copy(), best, baseline)

    Δ['Δ'] = Δ['Δ'].astype('float')
    return Δ
if ablation_plots : 
	save_as_pdf_pages([
	        plot_ablation_extended_Δ_ccdf(get_ablation_extended_Δ_df(table.copy(), M_sample_baseline=1, M_sample_path_IWVI=10, M_sample_path_VI=1, seed=s), 
	                vis_max_Δ=v, seed=s)
	    for s in [11, 12, 77] for v in [4] 
	], filename="ablation_extended_ccdf.pdf", path="../data/experiments/figures")
# exit()
################################################################################
########        Generate Path Figures from final metric    #########
################################################################################
def insert_at_ind(l, i, x):
	was_tuple = False
	if isinstance(l, tuple):
		l = list(l)
		was_tuple = True
	assert(isinstance(i, int))
	if i >= -1 :
		l.insert(i, x)
	elif i == -1:
		l.append(x)
	elif i<-1:
		l.insert(i+1, x)
	if was_tuple:
		return tuple(l)
	else :
		return l
def get_pair_Δ_df(table, apple1_name, apple1_index, apple2_name, apple2_index, max_Δ = 100.0, seed_pos = -1):
    unique_seeds = np.unique([v[seed_pos] for v in table.columns.values])

    Δ = pd.DataFrame(columns = ['tag', 'Δ'])
    Δ = pd.concat([
                        pd.DataFrame(
                        data = np.array([
                                        [str(apple2_name)+' - '+str(s)]*30 ,
                                        (table[insert_at_ind(apple2_index, seed_pos, s)] - 
                                         table[insert_at_ind(apple1_index, seed_pos, s)]).fillna(0).apply(lambda x: np.minimum(x, max_Δ))
                                        ]).T,
                        columns = Δ.columns) 
                        for s in unique_seeds
                        ], 
                         ignore_index= True)
    Δ['Δ'] = Δ['Δ'].astype('float')
    return Δ

def plot_Δ_pairwise_cdf(Δ, apple1_name, vis_max_Δ):
    p = (ggplot(Δ)
         +stat_ecdf(mapping=aes(x = 'Δ', color = 'tag', size = 'tag', alpha = 'tag'), geom='step', 
                    position='identity',na_rm=False)
         +scale_y_continuous(breaks = np.linspace(0, 1, 6), minor_breaks = 5)
         +scale_x_continuous(breaks = np.linspace(0, vis_max_Δ, 6), minor_breaks = 3)

         +coord_cartesian(xlim=[-0.5, vis_max_Δ])
         +theme(
                panel_grid=element_line(color="lightgray"),
                figure_size=(4.3, 4), 
               )
         +labs(x="$\Delta$ (in nats)\n $\Delta \coloneqq$ Lower bound improvement over "+str(apple1_name), 
               y="Frac. of models with more than $\Delta$ improvement",
               title = "Complementary CDF for Improvement, $\Delta$",
              )
         +scale_size_manual(np.array([2,2,2])*0.75)
         +scale_color_manual([ "#56B4E9","#D55E00", "#009E73"])
         +scale_alpha_manual([0.60]*5)
        )
    return p

def plot_Δ_pairwise_ccdf(Δ, apple1_name, apple2_name, vis_max_Δ = 5.0, vis_min_Δ = -0.5, vis_step = 1, x_axis_title_size = 13, y_axis_title_size = 12.8):
    p = plot_Δ_pairwise_cdf(Δ.copy(), apple1_name, vis_max_Δ)
    p._build()
    
    # prepare the dfp for ccdf
    dfp = p.layers[0].data
    dfp['y'] = 1 - dfp['y']
    dfp['x'] = pd.to_numeric(dfp['x'])
    dfp['y'] = pd.to_numeric(dfp['y'])
    for g in dfp['group'].unique():
        d = dfp[dfp['group'] == g].iloc[0, :]
        d['x'] = -1000.0
        d['y'] = 1.0
        dfp = dfp.append(d, ignore_index = True)
        d['x'] = 1000.0
        d['y'] = 0.0
        dfp = dfp.append(d, ignore_index = True)

    dfp['group'] = dfp['group'].astype('str')
    dfp['x'] = pd.to_numeric(dfp['x'])
    dfp['y'] = pd.to_numeric(dfp['y'])
    
    # plot ccdf
    p = (ggplot(dfp)
    +scale_y_continuous(breaks = np.linspace(0, 1, 6), minor_breaks = 5)
    +scale_x_continuous(breaks = np.arange(0, vis_max_Δ+1, vis_step) if vis_min_Δ >=-0.5 else np.arange(vis_min_Δ, vis_max_Δ+1, vis_step) , minor_breaks = 1) 
    +geom_step(mapping = aes(x = 'x', y = 'y', color = 'group', alpha = 'group', size = 'PANEL')) 
    +coord_cartesian(xlim=[vis_min_Δ, vis_max_Δ], ylim = [0, 1.0], expand = False)
    +theme(
            figure_size=(3.55, 3.45), 

            panel_grid=element_line(color="lightgray"),
#             panel_background=element_rect(fill= "white"),
            panel_border = element_rect(color = 'black'),
#             panel_grid_major = element_blank(),
#             panel_grid_minor = element_blank(),

            plot_title=element_blank(),
            plot_margin=0,
            
            axis_text=element_text(size = 13),
            axis_ticks_minor_x=element_line(color = "black"),
            axis_ticks_minor_y=element_line(color = "black"),
            axis_title_x=element_text(size = x_axis_title_size),
            axis_title_y=element_text(size = y_axis_title_size),
            legend_box_spacing=0,
            legend_margin=0,
            legend_spacing=0,
            legend_position=(0.625,0.945),
            legend_entry_spacing_y=0,
            legend_entry_spacing_x=20, 
            legend_background=element_rect(color = 'black', size = 0.5),
            legend_box_margin=0,
            legend_title=element_blank(),
            legend_text=element_text(size = 12.5),
        
       )
    +guides(
#      color=guide_legend(keywidth = 30, keyheight =10, nrow = 1, override_aes = {'size':2.5, 'alpha':0.7} ), 
     size=guide_legend(keywidth = 0, keyheight =0), 
    )
    +labs(x="$\Delta\coloneqq$  Method"+str(apple2_name)+" - "+str(apple1_name) +  " (nats)", 
        y="Frac. models with $\Delta$ improvement or more",
      color = 'Colors correspond to independent trials',
      alpha = 'Colors correspond to independent trials',
      size = 'Colors correspond to independent trials',
      )
#     +scale_color_manual([ "#56B4E9","#D55E00", "#009E73"], labels = ('', '', ''))
    +scale_color_manual([ "#D55E00", "#009E73","#0072B2"], guide = False)
    +scale_alpha_manual([0.55]*3, guide= False)
    +scale_size_continuous(range = [0.76,2.5], labels = ('Colors denote independent trials',))
    )
    return p


def easy_pairwise_plots(apple1, appl2, vis_max_Δ = 5.0, vis_min_Δ = -0.5, vis_step = 1.0, x_axis_title_size = 15, y_axis_title_size = 15, seed_pos = -2):
    Δ = get_pair_Δ_df(table.copy(), apple1_name = apple1[0], apple1_index=apple1[1:] , apple2_name = apple2[0], apple2_index = apple2[1:], seed_pos = seed_pos)
    p = plot_Δ_pairwise_ccdf(Δ, apple2_name = apple2[0], apple1_name=apple1[0], vis_max_Δ=vis_max_Δ, vis_min_Δ = vis_min_Δ, vis_step = vis_step, x_axis_title_size = x_axis_title_size, y_axis_title_size = y_axis_title_size)
    return p

if pair_plots:
	pairwise_plots = []
	apple1 = ("ADVI", "Full-rank Gaussian", 'closed-form-entropy', 0, 1, 1, "ADVI")
	apple2 = ("(0)", "Full-rank Gaussian", 'closed-form-entropy', 0, 1, 1, 'comprehensive step search')
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_max_Δ = 10, vis_min_Δ= -2.0, vis_step = 2.0))
	# vi_fam, grad_estimator_type, LI use, M_train, M_eval, seed(placed automatically)
	apple1 = ("Method (1)", "Full-rank Gaussian", 'IWAEDREG', 0, 1, 1, "comprehensive step search")
	apple2 = (" (3a)", "Full-rank Gaussian", 'IWAEDREG', 0, 1, 10, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2))
	# vi_fam, grad_estimator_type, LI use, M_train, M_eval, seed(placed automatically)
	apple1 = ("Method (0)", "Full-rank Gaussian", 'closed-form-entropy', 0, 1, 1, 'comprehensive step search')
	apple2 = (" (4a)", "Real NVP flows", 'IWAE', 0, 1, 1, 'comprehensive step search')
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_min_Δ=-1, vis_step=1))
	# vi_fam, grad_estimator_type, LI use, M_train, M_eval, seed(placed automatically)
	apple1 = ("Method (4a)", "Real NVP flows", 'IWAE', 0, 1, 1, 'comprehensive step search')
	apple2 = (" (4b)", "Real NVP flows", 'IWAEDREG', 0, 1, 1, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2))
	# vi_fam, grad_estimator_type, LI use, M_train, M_eval, seed(placed automatically)
	apple1 = ("Method (4b)", "Real NVP flows", 'IWAEDREG', 0, 1, 1, "comprehensive step search")
	apple2 = (" (4c)", "Real NVP flows", 'IWAEDREG', 0, 1, 10, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2))
	# vi_fam, grad_estimator_type, LI use, M_train, M_eval, seed(placed automatically)
	apple1 = ("Method (1)", "Full-rank Gaussian", 'IWAEDREG', 0, 1, 1, "comprehensive step search")
	apple2 = (" (2)", "Full-rank Gaussian", 'IWAEDREG', 1, 1, 1, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_min_Δ= -2.0))
	# vi_fam, grad_estimator_type, LI use, M_train, M_eval, seed(placed automatically)
	apple1 = ("Method (3a)", "Full-rank Gaussian", 'IWAEDREG', 0, 1, 10, "comprehensive step search")
	apple2 = (" (3b)", "Full-rank Gaussian", 'IWAEDREG', 0, 10, 10, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_min_Δ= -2.0))
	# vi_fam, grad_estimator_type, LI use, M_train, M_eval, seed(placed automatically)
	apple1 = ("Method (4c)", "Real NVP flows", 'IWAEDREG', 0, 1, 10, "comprehensive step search")
	apple2 = (" (4d)", "Real NVP flows", 'IWAEDREG', 0, 10, 10, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_min_Δ=-2.0))

	apple1 = ("Method (3b)$^{\\text{IWAE}\\nabla_{\\phi}}$", "Full-rank Gaussian", 'IWAE', 0, 10, 10, 'comprehensive step search')
	apple2 = (" (3b)$^{\\text{DReG}\\nabla_{\\phi}}$", "Full-rank Gaussian", 'IWAEDREG', 0, 10, 10, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_min_Δ= -1.0, x_axis_title_size=12))

	apple1 = ("Method (3b)$^{\\text{IWAE}\\nabla_{\\phi}}_\\text{w/ LI}$", "Full-rank Gaussian", 'IWAE', 1, 10, 10, 'comprehensive step search')
	apple2 = (" (3b)$^{\\text{DReG}\\nabla_{\\phi}}_\\text{w/ LI}$", "Full-rank Gaussian", 'IWAEDREG', 1, 10, 10, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_min_Δ= -1.0, x_axis_title_size=12))

	apple1 = ("Method (4d)$^{\\text{IWAE}\\nabla_{\\phi}}$", "Real NVP flows", 'IWAE', 0, 10, 10, 'comprehensive step search')
	apple2 = (" (4d)$^{\\text{DReG}\\nabla_{\\phi}}$", "Real NVP flows", 'IWAEDREG', 0, 10, 10, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_min_Δ= -1.0, x_axis_title_size=12))


	apple1 = ("ADVI", "Full-rank Gaussian", 'closed-form-entropy', 0, 1, 1, "ADVI")
	apple2 = (" (1)", "Full-rank Gaussian", 'IWAEDREG', 0, 1, 1, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_min_Δ= -1.0, x_axis_title_size=12))

	apple1 = ("Total Gradient", "Full-rank Gaussian", 'IWAE', 0, 1, 1, 'comprehensive step search')
	apple2 = (" (1)", "Full-rank Gaussian", 'IWAEDREG', 0, 1, 1, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_min_Δ= -1.0, x_axis_title_size=12))

	apple1 = ("Method (0)", "Full-rank Gaussian", 'closed-form-entropy', 0, 1, 1, 'comprehensive step search')
	apple2 = (" (1)", "Full-rank Gaussian", 'IWAEDREG', 0, 1, 1, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_min_Δ= -1.0, x_axis_title_size=12))


	apple1 = ("Method (0)", "Full-rank Gaussian", 'closed-form-entropy', 0, 1, 1, 'comprehensive step search')
	apple2 = (" (1)", "Full-rank Gaussian", 'IWAEDREG', 0, 1, 1, "comprehensive step search")
	pairwise_plots.append(easy_pairwise_plots(apple1, apple2, vis_min_Δ= -1.0, x_axis_title_size=12))


	save_as_pdf_pages(pairwise_plots, filename="pairwise_comparisons.pdf", path="../data/experiments/figures")
