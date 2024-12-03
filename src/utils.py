
import re
import time
import os
from datetime import datetime
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pickle
import random
from copy import deepcopy
import hypotheses
import math

from scipy.stats import ttest_1samp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,precision_score, \
                        recall_score,classification_report, \
                        accuracy_score, f1_score, log_loss, \
                       confusion_matrix, ConfusionMatrixDisplay,\
                          roc_auc_score, matthews_corrcoef, average_precision_score
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN,Birch,MeanShift, SpectralClustering

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterSampler

#import dimension reduction modules
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_random_seed(seed): #use for torch nn training in MC simulation
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False  

def set_output_dir(outpath):
    os.makedirs(outpath, exist_ok=True)
    return outpath

def purity_score(y_true, y_pred):
    # Matrix of contingency
    contingency_matrix = np.zeros((len(set(y_true)), len(set(y_pred))))
    for i, label in enumerate(y_true):
        contingency_matrix[label, y_pred[i]] += 1
    # Take the max label count for each cluster, sum them, and divide by total samples
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def get_metrics_of_hyperparm_set(y_preds):
    accuracies = []
    mcc_scores = []
    f1_scores = []
    roc_auc_scores = []
    pr_auc_scores = []

    # Iterate over all y_preds, assuming y_preds is a list of tuples (y_test, predicted)
    for y_test, predicted in y_preds:
        # Compute matches and row accuracies
        matches = (y_test == predicted)
        row_accuracies = np.mean(matches, axis=1)  # Mean across each row (for multi-label)
        
        # Compute overall accuracy for the current fold (mean of row accuracies)
        accuracies.append(np.mean(row_accuracies))

        # Compute MCC (Matthews Correlation Coefficient) for the current fold
        mcc = matthews_corrcoef(y_test.flatten(), predicted.flatten())  # Flatten if multi-label
        mcc_scores.append(mcc)

        # Compute F1 Score (for binary or multi-label)
        f1 = f1_score(y_test, predicted, average='macro')  # 'macro' averages F1 score across labels
        f1_scores.append(f1)

        # Compute AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
        auc_roc = roc_auc_score(y_test, predicted, average='macro', multi_class='ovr')  # 'ovr' for one-vs-rest
        roc_auc_scores.append(auc_roc)

        # Compute AUC-PR (Area Under the Precision-Recall Curve)
        auc_pr = average_precision_score(y_test, predicted, average='macro')  # 'macro' averages PR score across labels
        pr_auc_scores.append(auc_pr)

    # Calculate the average of all metrics across folds
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)  # Standard deviation for accuracy

    avg_mcc = np.mean(mcc_scores)
    avg_f1 = np.mean(f1_scores)
    avg_roc_auc = np.mean(roc_auc_scores)
    avg_pr_auc = np.mean(pr_auc_scores)
    return avg_accuracy, std_accuracy, avg_mcc, avg_f1, avg_roc_auc, avg_pr_auc



# Evaluation functions
def calculate_fin_metrics(y_pred, y_actual):
    num_samples, num_labels = y_pred.shape
    
    # Initialize lists to store individual metrics for each label
    label_accuracy = []
    auc_roc = []
    f1 = []
    mcc = []
    auprc = []
    
    for i in range(num_labels):
        # Calculate accuracy for each label
        acc = (y_pred[:, i] == y_actual[:, i]).mean()  # Mean accuracy for each label
        label_accuracy.append(acc)
        
        # Handle the AUC-ROC calculation only if both classes are present
        try:
            auc = roc_auc_score(y_actual[:, i], y_pred[:, i])
            auc_roc.append(auc)
        except ValueError:
            # Skip if only one class is present in y_true
            auc_roc.append(np.nan)
        
        # Calculate F1 score for each label
        f1_score_label = f1_score(y_actual[:, i], y_pred[:, i])
        f1.append(f1_score_label)
        
        # Calculate Matthews Correlation Coefficient (MCC) for each label
        mcc_score = matthews_corrcoef(y_actual[:, i], y_pred[:, i])
        mcc.append(mcc_score)
        
        # Calculate Average Precision (AU-PRC) for each label
        auprc_score = average_precision_score(y_actual[:, i], y_pred[:, i])
        auprc.append(auprc_score)
    
    # Overall accuracy for all 19 labels (mean of the per-label accuracies)
    overall_accuracy = np.mean(label_accuracy)
    
    return label_accuracy, auc_roc, f1, mcc, auprc, overall_accuracy

import pandas as pd
import numpy as np

def print_and_save_metrics(results, output_file):
    """
    Save and print the mean and standard deviation for each metric (excluding 'overall_accuracy')
    for each dataset and model.

    :param results: A dictionary containing dataset metrics with model names and their respective metric values.
    :param output_file: Path to the file where metrics will be saved.
    """
    print("hello")
    with open(output_file, 'w') as f:
        for dataset, models_metrics in results.items():
            f.write(f"Dataset: {dataset}\n")
            
            # Extracting the metrics for each model
            metrics_to_process = ['label_accuracy', 'auc_roc', 'f1', 'mcc', 'auprc']
            dataset_stats = {}
            model_stats = {}

            # Compute mean and std for each metric, per model
            for model_metrics in models_metrics:
                model_name = model_metrics["model"]
                model_stats[model_name] = {}

                for metric in metrics_to_process:
                    metric_values = model_metrics[metric]
                    mean_value = np.mean(metric_values)
                    std_value = np.std(metric_values)
                    model_stats[model_name][metric] = {"mean": mean_value, "std": std_value}
            
            # Now prepare the output for dataset-level statistics
            dataset_level_stats = {}

            for metric in metrics_to_process:
                # Gather all model mean values for the metric to calculate dataset-level mean and std
                model_mean_values = [model_stats[model][metric]["mean"] for model in model_stats]
                model_std_values = [model_stats[model][metric]["std"] for model in model_stats]

                dataset_level_stats[metric] = {
                    "mean": np.mean(model_mean_values),
                    "std": np.std(model_mean_values)
                }

            # Write out dataset-level stats
            f.write("Dataset-level statistics (mean and std of each model's metrics):\n")
            dataset_df = pd.DataFrame(dataset_level_stats).T
            dataset_df.index.name = 'Metric'
            f.write(dataset_df.to_string())
            f.write("\n\n")

            # Write out model-level stats (mean and std for each model and metric)
            f.write("Model-level statistics (mean and std for each model):\n")
            for model_name, stats in model_stats.items():
                f.write(f"Model: {model_name}\n")
                model_df = pd.DataFrame(stats).T
                model_df.index.name = 'Metric'
                f.write(model_df.to_string())
                f.write("\n\n")

            # Print the statistics to the console
            print(f"Metrics for dataset {dataset}:")
            print(dataset_df)
            for model_name, stats in model_stats.items():
                print(f"Model: {model_name}")
                print(pd.DataFrame(stats).T)

# Example usage
# print_and_save_metrics(results, 'metrics_summary.txt')
