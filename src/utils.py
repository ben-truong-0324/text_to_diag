
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
