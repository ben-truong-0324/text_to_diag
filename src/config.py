import socket

# Determine the hostname
hostname = socket.gethostname()
if hostname == "Khais-MacBook-Pro.local" or hostname == "Khais-MBP.attlocal.net":  # Replace with macbook hostname
    from config_dev import *  # Import everything from config_dev, small monte carlo count, smalle
else:
    from config_prod import * #BIG SIMU

import os


#define/create dir for outputs
def set_output_dir(outpath):
    os.makedirs(outpath, exist_ok=True)
    return outpath

EVAL_FUNC_METRIC = 'accuracy' #'f1' # 'accuracy' #for random srx implementation

AGGREGATED_OUTDIR = set_output_dir(f'{OUTPUT_DIR_A3}/ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/agregated_graphs')
CLUSTER_PKL_OUTDIR = set_output_dir(f'{OUTPUT_DIR_A3}/ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/pkl')
DREDUCED_CLUSTER_PKL_OUTDIR = set_output_dir(f'{OUTPUT_DIR_A3}/ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/pkl')
NN_CLUSTERED_DREDUCED_PKL_OUTDIR = set_output_dir(f'{OUTPUT_DIR_A3}/ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/cluster_of_reduced')
CLUSTER_GRAPH_OUTDIR = set_output_dir(f'{OUTPUT_DIR_A3}/ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/cluster')
DREDUCED_PKL_OUTDIR = set_output_dir(f'{OUTPUT_DIR_A3}/ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/dreduced_pickles')
TXT_OUTDIR = set_output_dir(f'{OUTPUT_DIR_A3}/ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/txt_stats')

OUTPUT_DIR_RAW_DATA_A3 = f'{OUTPUT_DIR_A3}/ver{DRAFT_VER_A3}/raw_data_assessment'
OUTPUT_DIR_CLUSTERING_BASELINE_A3 = f'{OUTPUT_DIR_A3}/ver{DRAFT_VER_A3}/baseline_cluster'


ALL_DREDUCED_USEFULNESS_WITH_NN_PICKLE_PATH = f'{OUTPUT_DIR_A3}/ver{DRAFT_VER_A3}_{EVAL_FUNC_METRIC}/dreduced_pickles/nn_dreduced_all_results.pkl'

K_FOLD_CV = 5  # Number of CV folds
DREDUCE_NUM = 5











roc_period = 10
window_size = 20
std_dev = 2
short_window = 12
long_window = 26
signal_window = 9
rsi_period = 14
pred_for_5d_delta = 1