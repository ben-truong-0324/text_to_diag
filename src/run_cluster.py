from config import *
from tests import *
import data_etl
import data_plots

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


from scipy.stats import ttest_1samp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,precision_score, \
                        recall_score,classification_report, \
                        accuracy_score, f1_score, log_loss, \
                       confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kf = KFold(n_splits=K_FOLD_CV, shuffle=True, random_state=GT_ID)

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


class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 32], dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        # Hidden layers with dropout in between
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)  # Apply dropout after each hidden layer
        
        return self.layers[-1](x)  # No activation or dropout on the output layer


def evaluate_model(model, X_val, y_val, device,criterion):
    model.eval()
    with torch.no_grad():
        # Ensure inputs are on the correct device
        X_val_tensor = X_val.to(device) if isinstance(X_val, torch.Tensor) else torch.tensor(X_val).to(device)
        y_val_tensor = y_val.to(device) if isinstance(y_val, torch.Tensor) else torch.tensor(y_val).to(device)
        
        # Forward pass
        outputs = model(X_val_tensor)
        
        # Calculate loss (for single-label classification)
        loss = criterion(outputs, y_val_tensor)
    return loss.item()

# Function to optimize the threshold for accuracy
def optimize_threshold(test_outputs_np, y_test_np):
    best_accuracy = 0
    best_threshold = 0.5  # Start with a default threshold (0.5)
    
    # Test a range of threshold values from 0 to 1
    for threshold in np.linspace(0, 1, 101):  # 101 points between 0 and 1
        predicted_labels = (test_outputs_np >= threshold).astype(int)
        
        # Calculate accuracy for this threshold
        label_accuracy = (predicted_labels == y_test_np).mean(axis=0)  # accuracy for each label
        accuracy = label_accuracy.mean()  # average accuracy across all labels
        
        # If this threshold gives a better accuracy, update the best threshold and accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    return best_threshold, best_accuracy

def train_nn_with_early_stopping_with_param(X_train, y_train, X_test, y_test, params, max_epochs=NN_MAX_EPOCH, patience=NN_PATIENCE):
    lr = params['lr']
    batch_size = params['batch_size']
    hidden_layers = params['hidden_layers']
    dropout_rate = params['dropout_rate']
    input_dim = X_train.shape[1]
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        # Multi-label classification (y_train has multiple labels per instance)
        output_dim = y_train.shape[1]  # Number of labels
    else:
        # Single-label classification (y_train has a single label per instance)
        output_dim = len(np.unique(y_train.cpu())) 
    model = SimpleNN(input_dim, output_dim, hidden_layers, dropout_rate=dropout_rate).to(device)



    if len(y_train.shape) == 1:  
        criterion = nn.CrossEntropyLoss()  # For single label classification (multi-class)
    else:  
        criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    epoch_trained = 0

    epoch_losses = []
    start_time = time.time()
    print("Starting training loop...")

    for epoch in range(max_epochs):
        epoch_trained+=1
        model.train()

        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)

        # Validation
        val_loss = evaluate_model(model, X_test, y_test, device,criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    runtime = time.time() - start_time
    print(f"Training completed in {runtime // 60:.0f}m {runtime % 60:.0f}s")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)

        _, predicted = torch.max(outputs, 1)

        if y_train.shape[1] > 1:
            test_outputs_np = outputs.sigmoid().cpu().numpy()  # Sigmoid for multi-label probability
           
            y_test_np = y_test.cpu().numpy()
            # predicted_labels = test_outputs_np.astype(int)
            best_threshold = .5
            # best_threshold, best_accuracy = optimize_threshold(test_outputs_np, y_test_np)
            predicted_labels = (test_outputs_np >= best_threshold).astype(int)
            # print(best_threshold)

            # Calculate accuracy, AUC-ROC, and F1-score for multi-label classification
            # Individual label accuracy (mean accuracy for each label)
            label_accuracy = (predicted_labels == y_test_np).mean(axis=0)
            accuracy = label_accuracy.mean()  # Average accuracy across all labels
            try:
                auc_roc = roc_auc_score(y_test_np, test_outputs_np, average="macro")
            except ValueError:
                auc_roc = float("nan")  # Handle cases where AUC-ROC can't be calculated
            f1 = f1_score(y_test_np, predicted_labels, average="macro")

        #if y is single label
        else:
            accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
            f1 = f1_score(y_test.cpu(), predicted.cpu(), average='weighted')

            # probs = torch.softmax(outputs, dim=1)
            # auc_roc = roc_auc_score(y_test.cpu(), probs.cpu(), multi_class='ovr')  # For multi-class problems
            probs = torch.sigmoid(outputs)[:, 0]  # Assuming the positive class is the first one
            auc_roc = roc_auc_score(y_test.cpu(), probs.cpu())

        ##################
    print(f"Training terminated after epoch {epoch_trained}, "
            f"Avg Label Accuracy: {accuracy:.4f}, "
            f"AUC-ROC: {auc_roc:.4f}, "
            f"F1-Score: {f1:.4f}")


    
    return accuracy, f1,auc_roc, runtime,model,epoch_losses


def run_clustering(cluster_algo, n_clusters, random_state, X, y):
    """
    Run a clustering algorithm (KMeans or GMM) on the given data.
    Parameters:
    - cluster_algo (str): The clustering algorithm to use 
    - n_clusters (int): The number of clusters to form (must be between 2 and 39).
    - random_state (int): The seed used by the random number generator.
    - X: pandas df, features
    - y: numpy arr, true label
    Returns:
    - runtime (float): Time taken to run the clustering algorithm.
    - labels (array): Cluster labels assigned to each data point.
    
    """
    print("running run_clustering ",cluster_algo)
    
    start_time = time.time()
    if cluster_algo == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
    elif cluster_algo == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    elif cluster_algo == 'aggclus':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif cluster_algo == 'dbscan':
        model = DBSCAN(eps=0.1, min_samples=2)
    elif cluster_algo == 'specclus':
        model = SpectralClustering(n_clusters=n_clusters, random_state=random_state)
    elif cluster_algo == 'birch':
        model = Birch(n_clusters=n_clusters)
    elif cluster_algo == 'meanshift':
        model = MeanShift()
    
     
    else:
        raise ValueError("Unsupported clustering algorithm. See run_clustering() for debug")
    print(X.shape)
    labels = model.fit_predict(X)
    runtime = time.time() - start_time
    return runtime, labels 


def collect_cluster_results(X, y, cluster_algo, preprocessing_tag = ""):
    """
    Collect cluster results by iterating n_clusters [2,40] if already ran, skip
    Parameters:
    - X: pandas df, features
    - y: numpy arr, true label
    - cluster_algo (str): The clustering algorithm to use
    Saves:
    - A pickle file containing the clustering results: runtime and labels.
    Raises:
    - ValueError: If an unsupported algorithm is provided.
    """
    
    outpath = f'{CLUSTER_PKL_OUTDIR}/{preprocessing_tag}{cluster_algo}_results.pkl'
    if not os.path.exists(outpath):
        results = {}
        for n_clusters in range(CLUSTERING_MIN_K, CLUSTERING_MAX_K):
            print("clustering features for ", n_clusters)
            random_state = np.random.randint(0, 10000)
            runtime, labels = run_clustering(cluster_algo, n_clusters, random_state, X, y)
            results[n_clusters] = {
                'runtime': runtime,
                'labels': labels,
            }
        with open(outpath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {outpath}")
    return outpath



def run_model_tuning_RO_for_Xy_srx_space(X, y, do_cv, random_opt_algo, best_overall_metric, best_overall_method, best_overall_model, best_overall_cv_losses, type_tag = ""):
    """
    Generalized helper method to run random optimization algorithms.
    
    Args:
        X: Feature data.
        y: Target labels.
        do_cv: Cross-validation flag.
        random_opt_algo: Optimization algorithm, e.g., "RHC", "GA", "SA".
        best_overall_metric: Initial or baseline metric to compare against.
        best_overall_method: Description of the method or optimization algorithm used.
        best_overall_model: Initial model or None.
    
    Returns:
        best_overall_metric: The best metric obtained.
        best_overall_model: The best model found during optimization.
        best_overall_method: The name of the best method.
        running_best_metrics_of_Xy: List of best metrics per iteration.
    """
    
    # Initialize the current best metric and model for this run
    running_best_model = None
    running_best_overall_cv_losses = None
    running_best_metrics_of_Xy_srx_space = []
    outer_ro_running_best_metric = 0
    #nested loops, outer loop iterates different params, inner loop iterates for kf CV

    # Set best metric and model placeholders for return
    final_best_metric = best_overall_metric
    final_best_model = best_overall_model
    final_best_method = best_overall_method
    final_best_overall_cv_losses = best_overall_cv_losses
    
    # Optimization loop with parameter sampling
    max_iterations = RANDOM_OPTIMIZATION_ITERATION_COUNT  

    rhc_restart_threshold = 5  # Number of iterations before restart
    rhc_no_improvement_count = 0  # Track no improvement iterations

    for i in range(RANDOM_OPTIMIZATION_ITERATION_COUNT):
        if random_opt_algo == "RHC":  # Random Hill Climbing
            current_params = deepcopy(RANDOM_SRX_PARAMS[np.random.randint(len(RANDOM_SRX_PARAMS))])
            for param_name in current_params.keys():
                if param_name == 'lr':
                    # Slightly adjust learning rate randomly
                    current_params[param_name] *= np.random.choice([0.9, 1.1])  

        elif random_opt_algo == "default":  #
            current_params = random.choice(RANDOM_SRX_PARAMS)

        current_metrics_of_Xy = []
        inner_cv_running_best_metric = 0

        avg_metric_per_cv = [0 for _ in range(K_FOLD_CV)] if do_cv else [0]
        cv_losses = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X) if do_cv else [(range(len(X)), range(len(X)))]):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train and evaluate model with current parameters
            accuracy, f1,auc_roc, runtime,temp_model,epoch_losses = train_nn_with_early_stopping_with_param(
                X_train, y_train, X_val, y_val, current_params
            )
            
            # Store the current metrics
            current_metrics_of_Xy.append((accuracy, f1, runtime))
            
            # Choose evaluation metric
            if "f1" in EVAL_FUNC_METRIC:
                avg_metric_per_cv[fold_idx] = f1
            elif "accuracy" in EVAL_FUNC_METRIC:
                avg_metric_per_cv[fold_idx] = accuracy
            elif "auc" in EVAL_FUNC_METRIC:
                avg_metric_per_cv[fold_idx] = auc_roc
            cv_losses.append(epoch_losses)

        # Calculate average metric across folds
        avg_metric = np.mean(avg_metric_per_cv)
        
        # Update running best if the new metric is better
        if avg_metric > inner_cv_running_best_metric:
            inner_cv_running_best_metric = avg_metric
            rhc_no_improvement_count = 0
        else:
            rhc_no_improvement_count += 1

        if inner_cv_running_best_metric > outer_ro_running_best_metric:
            outer_ro_running_best_metric = inner_cv_running_best_metric
            running_best_model = temp_model
            running_best_metrics_of_Xy_srx_space = current_metrics_of_Xy
            running_best_overall_cv_losses = cv_losses
        
        # Restart logic for RHC
        if rhc_no_improvement_count >= rhc_restart_threshold:
            print(f"Restarting RHC after {rhc_no_improvement_count} iterations without improvement.")
            rhc_no_improvement_count = 0  # Reset count
            inner_running_best_metric = 0  # Reset current best for new search

    # Check against overall best and update if necessary
    if outer_ro_running_best_metric > final_best_metric:
        final_best_metric = outer_ro_running_best_metric
        final_best_model = running_best_model
        final_best_method = type_tag
        final_best_overall_cv_losses = running_best_overall_cv_losses
            
    return final_best_metric, final_best_model, final_best_method, running_best_metrics_of_Xy_srx_space, final_best_overall_cv_losses



def get_cluster_usefulness_with_nn(results, X,y,outpath,cv_losses_outpath):
    if not os.path.exists(outpath):
        nn_results = {}
        running_best_model = None  
        running_best_cluster = None
        
        best_overall_model = None
        best_overall_method = None
        best_overall_metric = 0
        best_overall_cv_losses = None

        # Loop through each n_cluster and run the NN
        

        try:
            if len(y.shape) > 1 and y.shape[1] > 1:
                # Multi-label case: use FloatTensor for multi-hot encoded labels
                try:
                    y_labels = torch.FloatTensor(y).to(device)
                except:
                    y_labels = torch.FloatTensor(y.values).to(device)
            else:
                try:
                    y_labels = torch.LongTensor(y).to(device)
                except:
                    y_labels = torch.LongTensor(y.values).to(device)
        except AttributeError:
            # If y is a Pandas DataFrame or Series, convert to NumPy first
            y_values = y.values if hasattr(y, 'values') else y
            if len(y_values.shape) > 1 and y_values.shape[1] > 1:
                y_labels = torch.FloatTensor(y_values).to(device)
            else:
                y_labels = torch.LongTensor(y_values).to(device)
        # try:
        #     # For cases where y is already a NumPy array or similar structure
        #     y_labels = torch.FloatTensor(y).to(device)
        # except AttributeError:
        #     # If y is a Pandas DataFrame or Series, convert to NumPy first
        #     y_labels = torch.FloatTensor(y.values).to(device)

        for n_clusters in results.keys():
            labels = results[n_clusters]['labels']
            X_with_clustered_labels = pd.DataFrame(X)  # Assuming X_train is a DataFrame
            X_with_clustered_labels['cluster'] = labels  # Add cluster labels as a new feature
            X_features = torch.FloatTensor(X_with_clustered_labels.values).to(device)
            # best_overall_metric, best_cluster_model, best_cluster_count could be deprecated?
            best_overall_metric, best_overall_model, best_overall_method, running_metrics_Xy_srx_space, best_overall_cv_losses = run_model_tuning_RO_for_Xy_srx_space(
                X_features, 
                y_labels, 
                do_cv=True, 
                random_opt_algo="default", 
                best_overall_metric=best_overall_metric,  # Keyword argument
                best_overall_method=best_overall_method,    # Keyword argument
                best_overall_model=best_overall_model,    # Keyword argument
                best_overall_cv_losses = best_overall_cv_losses,
                type_tag=f"{n_clusters}_c"             # Keyword argument
            )
            nn_results[n_clusters] = {'mc_results': running_metrics_Xy_srx_space}

        #########################
        # Baseline without clustered labels
        X_baseline = torch.FloatTensor(X.values).to(device)  # Use original features


        # best_overall_metric, best_cluster_model, best_cluster_count could be deprecated?
        best_overall_metric, best_overall_model, best_overall_method, running_metrics_Xy_srx_space, best_overall_cv_losses = run_model_tuning_RO_for_Xy_srx_space(
            X_features, 
            y_labels, 
            do_cv=True, 
            random_opt_algo="default", 
            best_overall_metric=best_overall_metric,  # Keyword argument
            best_overall_method=best_overall_method,    # Keyword argument
            best_overall_model=best_overall_model,    # Keyword argument
            best_overall_cv_losses = best_overall_cv_losses,
            type_tag=f"baseline"             # Keyword argument
        )
        with open(cv_losses_outpath, 'wb') as f:
            pickle.dump(best_overall_cv_losses,f)
            
        nn_results["baseline"] = {'mc_results': running_metrics_Xy_srx_space}
        torch.save(best_overall_model, f'{AGGREGATED_OUTDIR}/best_{best_overall_method}.pth')
        with open(outpath, 'wb') as f:
            pickle.dump(nn_results, f)
        print("saved to ",outpath)

def get_p_value_if_monte_carlo_within_5_perc(cluster_nn_usefulness_results, outpath, outpath_pkl):
    if not os.path.exists(outpath):
        nn_statistics = {}
        for n_clusters, results in cluster_nn_usefulness_results.items():
            mc_results = np.array(results['mc_results'])  # Convert list of tuples to a numpy array
            accuracies, f1_scores, runtimes = mc_results[:, 0], mc_results[:, 1], mc_results[:, 2]

            # Calculate statistics
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)

            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)

            mean_runtime = np.mean(runtimes)
            std_runtime = np.std(runtimes)

            # Define bounds for the acceptable range (mean ± 5%)
            accuracy_bounds = (mean_accuracy * 0.95, mean_accuracy * 1.05)
            f1_bounds = (mean_f1 * 0.95, mean_f1 * 1.05)
            runtime_bounds = (mean_runtime * 0.95, mean_runtime * 1.05)

            # Perform one-sample t-test to check if means are within the bounds
            p_value_accuracy = ttest_1samp(accuracies, mean_accuracy)[1]
            p_value_f1 = ttest_1samp(f1_scores, mean_f1)[1]
            p_value_runtime = ttest_1samp(runtimes, mean_runtime)[1]
            

            # if  p_value_accuracy > .05 or p_value_f1 > .05 or p_value_runtime >.05:
            #     print("Learning model was not robust during MC simulation. Check .txt statistics for details")

            # Save statistics in the dictionary
            nn_statistics[n_clusters] = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'mean_runtime': mean_runtime,
                'std_runtime': std_runtime,
                'p_value_accuracy': p_value_accuracy,
                'p_value_f1': p_value_f1,
                'p_value_runtime': p_value_runtime,
            }

        # Print the results and save to a text file
        with open(outpath, 'w') as f:
            for n_clusters, stats in nn_statistics.items():
                f.write(f'n_clusters: {n_clusters}\n')
                for key, value in stats.items():
                    f.write(f'{key}: {value}\n')
                f.write('\n')  # Add a newline for better readability
        if outpath_pkl:
            with open(outpath_pkl, 'wb') as f:
                pickle.dump(nn_statistics, f)

def implement_clustering(X,y):
    #TODO: add 'none' into CLUSTER_ALGORITHMS to avoid repeated baseline calcs
    clustered_model_results = {}
    for cluster_algo in CLUSTER_ALGORITHMS:
        print(cluster_algo)
        cluster_saved_path = collect_cluster_results(X, y, cluster_algo, f"baseline{CLUSTERING_MAX_K}_") #clustering for c from 2 to CLUSTERING_MAX_K

        with open(cluster_saved_path, 'rb') as f:
            cluster_results = pickle.load(f)
        data_plots.make_cluster_graphs(cluster_results,X,
                            CLUSTER_GRAPH_OUTDIR, cluster_algo,f"{CLUSTERING_MAX_K}_")
        #############RUNNING NN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #to test for results using torch with cpu, run a differnt DRAFT_VER_A3
        clustered_nn_pkl_path = f'{CLUSTER_PKL_OUTDIR}/{CLUSTERING_MAX_K}_{cluster_algo}_cluster_as_usefulness_with_nn_wrapping_results.pkl'
        cv_losses_outpath = f'{CLUSTER_PKL_OUTDIR}/{CLUSTERING_MAX_K}_{cluster_algo}_cv_losses.pkl'
        get_cluster_usefulness_with_nn(cluster_results,X,y,clustered_nn_pkl_path,cv_losses_outpath)
        with open(clustered_nn_pkl_path, 'rb') as f:
            cluster_nn_usefulness_results = pickle.load(f)
        
        data_plots.plot_cluster_usefulness_by_nn_banded_mean(cluster_nn_usefulness_results, 
            f'{AGGREGATED_OUTDIR}/{CLUSTERING_MAX_K}_{cluster_algo}_usefulness_nn.png',)
        
########################################
from scipy.stats import kurtosis

def get_dimension_reduced_features(method, k_dimension, X, y, pickle_outpath):  
    if not os.path.exists(pickle_outpath):  
        print(f"{pickle_outpath} does not exist")
        if method == "PCA":
            model = PCA(n_components=k_dimension)
            X_reduced = model.fit_transform(X)
            # Get the eigenvalues (explained variance) of each principal component
            eigenvalues = model.explained_variance_
            print(f"PCA eigenvalues for k={k_dimension}: {eigenvalues}")
            with open(pickle_outpath, 'wb') as f:
                pickle.dump((X_reduced, y, eigenvalues), f)

        elif method == "ICA":
            model = FastICA(n_components=k_dimension)
            X_reduced = model.fit_transform(X)
            # Calculate kurtosis for each ICA component
            component_kurtosis = kurtosis(X_reduced, axis=0)
            print(f"ICA component kurtosis for k={k_dimension}: {component_kurtosis}")
            with open(pickle_outpath, 'wb') as f:
                pickle.dump((X_reduced, y, component_kurtosis), f)
        elif method == "RCA":
            model = GaussianRandomProjection(n_components=k_dimension)
            X_reduced = model.fit_transform(X)
            with open(pickle_outpath, 'wb') as f:
                pickle.dump((X_reduced, y), f)
        elif method == "RP":
            model = SparseRandomProjection(n_components=k_dimension)
            X_reduced = model.fit_transform(X)
            with open(pickle_outpath, 'wb') as f:
                pickle.dump((X_reduced, y), f)
        elif method == "LDA":
            model = LDA(n_components=k_dimension)
            X_reduced = model.fit_transform(X, y)  # LDA requires fit with X and y
            with open(pickle_outpath, 'wb') as f:
                pickle.dump((X_reduced, y), f)
        elif method == "RandomForest":
            rf = RandomForestClassifier(n_estimators=200)
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:k_dimension]  # Get the top k_dimension features
            X_reduced = X.iloc[:, indices]
            with open(pickle_outpath, 'wb') as f:
                pickle.dump((X_reduced, y), f)
        else:
            raise ValueError(f"Error at get_dimension_reduced_features().")

###################################
# Initialize the results dictionary

def get_dreduced_usefulness_with_nn(X, y, max_k_dimension,pickle_outpath,cv_losses_outpath):
    
    if not os.path.exists(pickle_outpath):  
        print("harro")
        nn_dreduced = {}
        running_best_model = None
        running_best_method = None
        best_cluster_model = None

        best_overall_method = None

        running_best_model = None  
        running_best_cluster = None
        
        best_overall_model = None
        best_cluster_count = None
        best_overall_metric = 0
        best_overall_cv_losses = None

        # try:
        #     y_labels = torch.LongTensor(y.values).to(device)
        # except:
        #     y_labels = torch.LongTensor(y.values).to(device)
        try:
            if len(y.shape) > 1 and y.shape[1] > 1:
                # Multi-label case: use FloatTensor for multi-hot encoded labels
                try:
                    y_labels = torch.FloatTensor(y).to(device)
                except:
                    y_labels = torch.FloatTensor(y.values).to(device)
            else:
                try:
                    y_labels = torch.LongTensor(y).to(device)
                except:
                    y_labels = torch.LongTensor(y.values).to(device)
        except AttributeError:
            # If y is a Pandas DataFrame or Series, convert to NumPy first
            y_values = y.values if hasattr(y, 'values') else y
            if len(y_values.shape) > 1 and y_values.shape[1] > 1:
                y_labels = torch.FloatTensor(y_values).to(device)
            else:
                y_labels = torch.LongTensor(y_values).to(device)

        for method in DIMENSION_REDUCE_METHODS:
            for k_dimension in range(int(max_k_dimension/2),max_k_dimension):
                try:
                    print(f"{method} {k_dimension}")

                    pickle_path = f'{DREDUCED_PKL_OUTDIR}/{method}_reduced_{k_dimension}_results.pkl'
                    with open(pickle_path, 'rb') as f:
                        X_reduced, y = pickle.load(f)
                    if "Random" in method:
                        X_reduced = X_reduced.values
                    X_reduced = torch.FloatTensor(X_reduced).to(device)  

                    # best_overall_metric, best_cluster_model, best_cluster_count could be deprecated?
                    best_overall_metric, best_overall_model, best_overall_method, running_metrics_Xy_srx_space,best_overall_cv_losses = run_model_tuning_RO_for_Xy_srx_space(
                        X_reduced, 
                        y_labels, 
                        do_cv=True, 
                        random_opt_algo="default", 
                        best_overall_metric=best_overall_metric,  # Keyword argument
                        best_overall_method=best_overall_method,    # Keyword argument
                        best_overall_model=best_overall_model,    # Keyword argument
                        best_overall_cv_losses = best_overall_cv_losses,
                        type_tag=f"{method}_{k_dimension}"             # Keyword argument
                    )
                    nn_dreduced[f"{method}_{k_dimension}"] = { 'mc_results': running_metrics_Xy_srx_space}

                except Exception as e:
                    if "LDA" not in method:
                        print(e)
        
        ######### get baseline
        # Baseline without reduced features
        X_baseline = torch.FloatTensor(X.values).to(device)  # Use original features

        # best_overall_metric, best_cluster_model, best_cluster_count could be deprecated?
        best_overall_metric, best_overall_model, best_overall_method, running_metrics_Xy_srx_space,best_overall_cv_losses = run_model_tuning_RO_for_Xy_srx_space(
                X_baseline, 
                y_labels, 
                do_cv=True, 
                random_opt_algo="default", 
                best_overall_metric=best_overall_metric,  # Keyword argument
                best_overall_method=best_overall_method,    # Keyword argument
                best_overall_model=best_overall_model,    # Keyword argument
                best_overall_cv_losses = best_overall_cv_losses,
                type_tag=f"baseline"             # Keyword argument
                )
        nn_dreduced["baseline"] = {'mc_results': running_metrics_Xy_srx_space}

        torch.save(best_overall_model, f'{AGGREGATED_OUTDIR}/best_m_{best_overall_method}.pth')

        # Save the nn_dreduced results to a pickle file
        with open(cv_losses_outpath, 'wb') as f:
            pickle.dump(best_overall_cv_losses,f)

        with open(pickle_outpath, 'wb') as f:
            pickle.dump(nn_dreduced, f)
        print("saved to ", pickle_outpath)
 
def implement_dimension_reduction(X,y):
    max_k_dimension = X.shape[1]-1
    for method in DIMENSION_REDUCE_METHODS:
        for k_dimension in range(int(max_k_dimension/2),max_k_dimension):
            # Adjust for LDA to not exceed the limit
            if method == "LDA":
                n_classes = len(np.unique(y))
                lda_max_k_dimension = min(X.shape[1], n_classes - 1)  # Max k for LDA
                if k_dimension > lda_max_k_dimension:
                    #print(f"Skipping LDA with k_dimension={k_dimension}, exceeds limit.")
                    continue  
            get_dimension_reduced_features(method, k_dimension, 
                X, y, 
                f'{DREDUCED_PKL_OUTDIR}/{method}_reduced_{k_dimension}_results.pkl')

    all_dreduced_usefulness_with_nn_pickle_path = f'{DREDUCED_PKL_OUTDIR}/nn_dreduced_all_results.pkl'
    cv_losses_outpath = f'{DREDUCED_PKL_OUTDIR}/cv_losses.pkl'

    get_dreduced_usefulness_with_nn(X, y, max_k_dimension, all_dreduced_usefulness_with_nn_pickle_path, cv_losses_outpath)
    with open(all_dreduced_usefulness_with_nn_pickle_path, 'rb') as f:
        all_dreduced_results = pickle.load(f)
    # print(all_dreduced_results)
    hypotheses.evaluate_dreduced_vs_baseline(all_dreduced_results, "first")
    print("done")
    data_plots.plot_dreduced_usefulness_by_nn_banded_mean(all_dreduced_results, 
        f'{AGGREGATED_OUTDIR}/DReduced_usefulness_nn.png',)
    data_plots.plot_dreduced_usefulness_by_nn_acc_f1(all_dreduced_results, 
        f'{AGGREGATED_OUTDIR}/DReduced_acc_f1_usefulness_nn.png',)


def calc_purity_score(X,y,len_unique_labels_multiple,purity_pkl_path,purity_txt_path):
    print("sup", len_unique_labels_multiple)
    if not os.path.exists(purity_pkl_path) or os.path.exists(purity_pkl_path) :
        print(purity_pkl_path)
        # Store purity scores for each configuration
        purity_scores = {}
        try:
            for method in DIMENSION_REDUCE_METHODS:
                for cluster_algo in CLUSTER_ALGORITHMS:
                    # Load the clustering results
                    baseline_pkl_path = f'{CLUSTER_PKL_OUTDIR}/baseline{CLUSTERING_MAX_K}_{cluster_algo}_results.pkl'
                    with open(baseline_pkl_path, 'rb') as f:
                        baseline_cluster_results = pickle.load(f)
                    # Compute purity score only for n_clusters == len_unique_labels_multiple
                    if len_unique_labels_multiple in baseline_cluster_results:
                        labels = baseline_cluster_results[len_unique_labels_multiple]['labels']
                        score = purity_score(y, labels)
                        
                    #     # Store the purity score
                        purity_scores[f'{method}_0d_{cluster_algo}_{len_unique_labels_multiple}clusters'] = score

        except Exception as e:
            if "LDA" not in method and "Forest" not in method: 
                print(f"Error processing {method}_0d_{cluster_algo}: {e}")
                #weird bug with randomforsest_1d
            else: print(e)


        # Iterate over methods, dimensions, and clustering algorithms
        for method in DIMENSION_REDUCE_METHODS:
            for k_dimension in range(1, X.shape[1] - 1):
                for cluster_algo in CLUSTER_ALGORITHMS:
                    try:
                        # Load the clustering results
                        pickle_path = f'{CLUSTER_PKL_OUTDIR}/{method}_{k_dimension}d_{cluster_algo}_results.pkl'
                        with open(pickle_path, 'rb') as f:
                            clustered_of_reduced_results = pickle.load(f)
                        
                        # Compute purity score only for n_clusters == len_unique_labels_multiple
                        if len_unique_labels_multiple in clustered_of_reduced_results:
                            labels = clustered_of_reduced_results[len_unique_labels_multiple]['labels']
                            score = purity_score(y, labels)
                            
                            # Store the purity score
                            purity_scores[f'{method}_{k_dimension}d_{cluster_algo}_{len_unique_labels_multiple}clusters'] = score

                    except Exception as e:
                        if "LDA" not in method and "Forest" not in method: 
                            print(f"Error processing {method}_{k_dimension}d_{cluster_algo}: {e}")
                            #weird bug with randomforsest_1d
        print("before dump")
        print(purity_scores)

        with open(purity_pkl_path, 'wb') as f:
            pickle.dump(purity_scores, f)
        with open(purity_txt_path, 'w') as f:
            for config, score in purity_scores.items():
                f.write(f"{config}: {score}\n")
        print(f"Purity scores saved to {purity_pkl_path} \nand {purity_txt_path}.")

def compile_all_pickles_to_one(big_pkl_path, X):
    if not os.path.exists(big_pkl_path):
        compiled_results = {}
        for method in DIMENSION_REDUCE_METHODS:
            compiled_results[method] = {}
            for k_dimension in range(1,X.shape[1]-1):
                compiled_results[method][k_dimension] = {}
                for cluster_algo in CLUSTER_ALGORITHMS:
                    pickle_path = f'{CLUSTER_PKL_OUTDIR}/{method}_{k_dimension}d_{cluster_algo}_results.pkl'
                    
                    # Check if the pickle file exists and load its data
                    if os.path.exists(pickle_path):
                        with open(pickle_path, 'rb') as f:
                            clustered_results = pickle.load(f)
                        # Store clustered results in the compiled dictionary
                        compiled_results[method][k_dimension][cluster_algo] = clustered_results
                    else:
                        print(f'Pickle file not found: {pickle_path}')
        # Save the compiled results to a single big pickle file
        with open(big_pkl_path, 'wb') as f:
            pickle.dump(compiled_results, f)
        print(f'Compiled results saved to: {big_pkl_path}')




def generate_all_pickles_into_nn_training_datasets(compiled_pkl_path, output_pkl_path, X_original, ):
    # Load compiled pickle file
    if not os.path.exists(output_pkl_path):
        with open(compiled_pkl_path, 'rb') as f:
            compiled_results = pickle.load(f)
        nn_datasets = {}
        for dreduc_algo, dreduc_data in compiled_results.items():
            for k_dim, kdim_data in dreduc_data.items():
                # Retrieve X reduced of dreduc_algo k_dim
                reduced_features_pickle_path = f'{DREDUCED_PKL_OUTDIR}/{dreduc_algo}_reduced_{k_dim}_results.pkl'
                
                try:
                    with open(reduced_features_pickle_path, 'rb') as f:
                        X_reduced, _ = pickle.load(f)  # We already have y, so ignore it from the loaded pickle
                except: 
                    print(f'Warning: Reduced features file not found: {reduced_features_pickle_path}')
                    continue
                
                if isinstance(X_reduced, np.ndarray):
                    X_reduced = pd.DataFrame(X_reduced)

                for cluster_algo, cluster_data in kdim_data.items():
                    for n_clusters, cluster_info in cluster_data.items():
                        # Make a copy of X_reduced and add clustered labels
                        X_clustered_reduced = X_reduced.copy()
                        X_clustered_reduced['clustered_label'] = cluster_info['labels']
                        
                        # Store the dataset with unique identifier
                        nn_datasets[(dreduc_algo, k_dim, cluster_algo, n_clusters)] = {
                            'X_clustered_reduced': X_clustered_reduced,
                        }

        # Save the datasets to a new pickle file


        with open(output_pkl_path, 'wb') as f:
            pickle.dump(nn_datasets, f)
        print(f'Training datasets for NN saved to: {output_pkl_path}')


def get_clustered_reduced_usefulness_with_nn(big_nn_input_pkl_path,X, y, big_nn_output_pkl_path, big_nn_output_txt_path, cv_losses_outpath):
    if not os.path.exists(big_nn_output_pkl_path):  
        nn_clustered_dreduced = {}
    
        best_overall_metric = 0
        best_overall_cv_losses = None
        best_overall_model = None
        best_overall_method = None
        running_metrics_Xy_srx_space = None

        # try:
        #     y_labels = torch.LongTensor(y.values).to(device)
        # except:
        #     y_labels = torch.LongTensor(y.values).to(device)
        try:
            if len(y.shape) > 1 and y.shape[1] > 1:
                # Multi-label case: use FloatTensor for multi-hot encoded labels
                try:
                    y_labels = torch.FloatTensor(y).to(device)
                except:
                    y_labels = torch.FloatTensor(y.values).to(device)
            else:
                try:
                    y_labels = torch.LongTensor(y).to(device)
                except:
                    y_labels = torch.LongTensor(y.values).to(device)
        except AttributeError:
            # If y is a Pandas DataFrame or Series, convert to NumPy first
            y_values = y.values if hasattr(y, 'values') else y
            if len(y_values.shape) > 1 and y_values.shape[1] > 1:
                y_labels = torch.FloatTensor(y_values).to(device)
            else:
                y_labels = torch.LongTensor(y_values).to(device)

        # Baseline without reduced features
        X_baseline = torch.FloatTensor(X.values).to(device)  # Use original features
        # best_overall_metric, best_cluster_model, best_cluster_count could be deprecated?
        best_overall_metric, best_overall_model, best_overall_method, running_metrics_Xy_srx_space,best_overall_cv_losses = run_model_tuning_RO_for_Xy_srx_space(
                X_baseline, 
                y_labels, 
                do_cv=True, 
                random_opt_algo="default", 
                best_overall_metric=best_overall_metric,  # Keyword argument
                best_overall_method=best_overall_method,    # Keyword argument
                best_overall_model=best_overall_model,    # Keyword argument
                best_overall_cv_losses = best_overall_cv_losses,
                type_tag=f"baseline"             # Keyword argument
                )
        nn_clustered_dreduced["baseline"] = {'mc_results': running_metrics_Xy_srx_space}

        with open(big_nn_input_pkl_path, 'rb') as f:
            big_nn_dataset = pickle.load(f)
        for (dreduc_algo, k_dim, cluster_algo, n_clusters), data in big_nn_dataset.items():
            X_reduced = data['X_clustered_reduced']
            X_reduced = torch.FloatTensor(X_reduced.values).to(device)
            print((dreduc_algo, k_dim, cluster_algo, n_clusters))

            best_overall_metric, best_overall_model, best_overall_method, running_metrics_Xy_srx_space,best_overall_cv_losses = run_model_tuning_RO_for_Xy_srx_space(
                X_baseline, 
                y_labels, 
                do_cv=True, 
                random_opt_algo="default", 
                best_overall_metric=best_overall_metric,  # Keyword argument
                best_overall_method=best_overall_method,    # Keyword argument
                best_overall_model=best_overall_model,    # Keyword argument
                best_overall_cv_losses = best_overall_cv_losses,
                type_tag=f"{dreduc_algo}_{k_dim}k_{cluster_algo}_{n_clusters}c",            # Keyword argument
                )
            nn_clustered_dreduced[(dreduc_algo, k_dim, cluster_algo, n_clusters)] = {
                'mc_results': running_metrics_Xy_srx_space }
        torch.save(best_overall_model, f'{AGGREGATED_OUTDIR}/best_model_{best_overall_method}.pth')
        # loaded_model = torch.load('best_model.pth')

        with open(cv_losses_outpath, 'wb') as f:
            pickle.dump(best_overall_cv_losses,f)


        with open(big_nn_output_pkl_path, 'wb') as f:
            pickle.dump(nn_clustered_dreduced, f)
        print(f'NN results saved to: {big_nn_output_pkl_path}')

    if not os.path.exists(big_nn_output_txt_path):  
        if not nn_clustered_dreduced:
            with open(big_nn_output_pkl_path, 'rb') as f:
                nn_clustered_dreduced = pickle.load(f)
        with open(big_nn_output_txt_path, 'w') as f:
            for config, score in nn_clustered_dreduced.items():
                f.write(f"{config}: {score}\n")
        print(f'NN results saved to: {big_nn_output_txt_path}')

def implement_clustering_on_reduced_features(X,y):
    print("ey")
    all_dreduced_usefulness_with_nn_pickle_path = ALL_DREDUCED_USEFULNESS_WITH_NN_PICKLE_PATH
    for cluster_algo in CLUSTER_ALGORITHMS:
        collect_cluster_results(X, y, cluster_algo, "baseline_") #clustering for c from 2 to CLUSTERING_MAX_K

    # do clustering for reduced feature sets, instead of raw feature sets
    for method in DIMENSION_REDUCE_METHODS:
        for k_dimension in range(1,X.shape[1]-1):
            try: #try because some dimension reduction like LDA has different dimension count
                pickle_path = f'{DREDUCED_PKL_OUTDIR}/{method}_reduced_{k_dimension}_results.pkl'
                with open(pickle_path, 'rb') as f:
                    X_reduced, y = pickle.load(f)
                for cluster_algo in CLUSTER_ALGORITHMS:
                    collect_cluster_results(X_reduced, y, cluster_algo, f'{method}_{k_dimension}d_') #if already ran, will skip
            except: 
                if EXP_DEBUG:
                    print(f"{method} {k_dimension} failed to cluster at implement_clustering_on_reduced_features() ")
    
    #get some graphs for clustered and reduced, grouped by cluster method and reduction method, and their own k and n values
    # for method in DIMENSION_REDUCE_METHODS:
    #     for k_dimension in range(1,X.shape[1]-1):
    #         for cluster_algo in CLUSTER_ALGORITHMS:
    #             try:
    #                 pickle_path = f'{DREDUCED_CLUSTER_PKL_OUTDIR}/{method}_{k_dimension}d_{cluster_algo}_results.pkl'
    #                 with open(pickle_path, 'rb') as f:
    #                     clustered_of_reduced_results = pickle.load(f)

    #                 data_plots.make_cluster_of_reduced_graphs(clustered_of_reduced_results, X,
    #                     NN_CLUSTERED_DREDUCED_PKL_OUTDIR, cluster_algo,
    #                     f'{method}_{k_dimension}d_')
    #             except: pass
    
    ####### for purity score of c cluster as multiples of target label count
    unique_labels, counts = np.unique(y, return_counts=True)

    multiples_list = [i * len(unique_labels) for i in range(1, CLUSTERING_MAX_K // len(unique_labels) )]
    #by clustering c
    print(multiples_list)
    for len_unique_labels_multiple in multiples_list:
        purity_pkl_path = f'{CLUSTER_PKL_OUTDIR}/{len_unique_labels_multiple}_cluster_purity_scores.pkl'
        purity_txt_path = f'{TXT_OUTDIR}/{len_unique_labels_multiple}_cluster_purity_scores.txt'
        calc_purity_score(X,y,len_unique_labels_multiple, purity_pkl_path, purity_txt_path)

        # Load the purity scores from the .pkl file and call the function
        with open(purity_pkl_path, 'rb') as f:
            purity_scores = pickle.load(f)
        # data_plots.plot_purity_score_of_c_cluster_same_as_original_targets(purity_scores, AGGREGATED_OUTDIR, f'{len_unique_labels_multiple}clusters_')
        data_plots.plot_purity_score_of_c_cluster(purity_scores, AGGREGATED_OUTDIR, f'{len_unique_labels_multiple}_seperate_clusters_')
        print("donezo")
        for thres in np.arange(.001,.1,.001):
            hypotheses.run_dred_improves_purity_score_hypo_test(purity_scores, thres) #test of improved 10%
        # hypotheses.run_dred_improves_purity_score_hypo_test(purity_scores, .01) #test if improved 1%

        data_plots.plot_purity_significance_from_pkl(AGGREGATED_OUTDIR)
                
    ######### now get usefulness by training nn
    big_pkl_path = f'{DREDUCED_CLUSTER_PKL_OUTDIR}/agregated_clustered_reduced_results.pkl'
    big_nn_input_pkl_path = f'{DREDUCED_CLUSTER_PKL_OUTDIR}/nn_training_data_agregated_clustered_reduced_results.pkl'

    compile_all_pickles_to_one(big_pkl_path,X)
    generate_all_pickles_into_nn_training_datasets(big_pkl_path, big_nn_input_pkl_path, X)
    #stored as dict with key ('PCA', 1, 'kmeans', 2) or nn_datasets[(dreduc_algo, k_dim, cluster_algo, n_clusters)] = {'X_clustered_reduced': X_clustered_reduced, }
    big_nn_output_pkl_path = f'{DREDUCED_CLUSTER_PKL_OUTDIR}/nn_accuracy_f1_runtime_agregated_clustered_reduced_results.pkl'
    big_nn_output_txt_path = f'{TXT_OUTDIR}/nn_accuracy_f1_runtime_clustered_reduced_results.txt'
    cv_losses_outpath = f'{DREDUCED_CLUSTER_PKL_OUTDIR}/clustered_reduced_cv_losses.pkl'

    get_clustered_reduced_usefulness_with_nn(big_nn_input_pkl_path,X, y, 
                    big_nn_output_pkl_path,big_nn_output_txt_path, cv_losses_outpath)
    #get monte carlo average and stats
    with open(big_nn_output_pkl_path, 'rb') as f:
        clustered_reduced_results = pickle.load(f)

    #each PCA, kmeans pair makes a 3d graph
    data_plots.plot_3d_comparison(clustered_reduced_results, color_map="plasma", 
                                outpath=f'{AGGREGATED_OUTDIR}')

    data_plots.plot_multi_histograms(clustered_reduced_results, outpath=f'{AGGREGATED_OUTDIR}')


    #show that for each PCA, as kmeans k varies the distribution tightness change? more tail end big std big misses vs more consistent small std



    # big_nn_mc_stats_output_pkl_path = f'{NN_CLUSTERED_DREDUCED_PKL_OUTDIR}/nn_stats_accuracy_f1_runtime_agregated_clustered_reduced_results.pkl'
    # big_nn_mc_stats_output_txt_path = f'{TXT_OUTDIR}/mc_nn_accuracy_f1_runtime_clustered_reduced_results.txt'
    # get_p_value_if_monte_carlo_within_5_perc(clustered_reduced_results, big_nn_mc_stats_output_txt_path,
    # big_nn_mc_stats_output_pkl_path,)

    # with open(big_nn_mc_stats_output_pkl_path, 'rb') as f:
    #     nn_statistics = pickle.load(f)
    # data_plots.plot_3d_comparison(nn_statistics, metric="accuracy", color_map="plasma")
    #change this from scatter plot to heat map where higher peaks are green and lower are red. change to only 1 value per x-y
    #x 0 y 0 means baseline or no DRed and no clustering applied

    #need 12 graphs
    #one is PCA, Kmeans, dataset1
    #       PCA, GMM, dataset1

#call this separately if starting on new dataset
def check_etl():
    X, y = data_etl.get_data(DATASET_SELECTION,1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=GT_ID)
    test_data_etl_input_check(X, y, X_train, X_test, y_train, y_test, verbose = 1)
    data_etl.graph_raw_data(X, y)
    TestClusteringFunctions()
    print("======> Data verification complete")
    return X,y,X_train, X_test, y_train, y_test 

###############
def main(): 
    np.random.seed(GT_ID)
    
    X,y,X_train, X_test, y_train, y_test  = check_etl()
    #add expedient training size minimum needed to plateau 

    #model_tuning = 1,     
    implement_clustering(X, y)
    implement_dimension_reduction(X,y)
    implement_clustering_on_reduced_features(X,y)


    #if we have unga_bunga, what is the search space scope still?
    #how can human intervention guide unga_bunga to a different search_space?

    #3 different kinds of human insight-driven intervevtion:
    #sigma tightness: what to do to make outputs more consistent
    #accuracy, f1,...: what to do to improve baseline
    #

    #srx space: [ETL, DReduction, Clustering, Manual Fts, Model choice, Model RO, Eval Func Metric]
    #eval_func: sigma tightness and expected value as composite index?

    #for different features added to make a new feature set?
    #for different model architecture and their own params?

    #feature to use best_model for inference tasks daily
    
if __name__ == "__main__":
    print(f"Torch will be running on {device}")
    # check_etl()
    main()
    

