
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from time import perf_counter
import os

import data_etl

from sklearn.model_selection import train_test_split
from config import *
from sklearn.neural_network import MLPClassifier
import mlrose_hiive as mlrose




def run_nn_validation_experiment():
    output_dir = f'{OUTPUT_DIR_OPTIMIZE}/ver{OPT_DRAFT_VER}nn'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    # Set range for training set sizes

    hidden_nodes_options = [(8,), 
    (16,), (32,), 
    # (64,), 
    # (128,), 
    (8, 8), 
    # (8, 16), 
    # (8, 32),
    # (8, 64),
    ]  # List of hidden nodes configurations
    learning_rates = [
        0.01, 
        0.05,
        0.1, 
        0.25,
        # 0.5,
        ]  # List of learning rates
    activation_functions = ['relu', 
    'sigmoid', 'tanh'
    ]  # List of activation functions

    results = {
        'fit_times': {},
        'accuracies': {},
        'f1_scores': {}
    }

    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent']

    # Load and split the dataset
    X_df, Y_df = data_etl.get_data("credit", 1, 0)
    X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, train_size=TRAIN_SIZE, random_state=GT_ID)

    # Optimization Algorithms
    for algo in algorithms:
        for nodes in hidden_nodes_options:
            for lr in learning_rates:
                for activation in activation_functions:
                    key = f'{algo} {lr} lr {nodes[0]} nodes {activation} activation_func'
                    print(key)
                    # Initialize lists if key doesn't exist
                    
                    clf_opt = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=activation, algorithm=algo, 
                                                max_iters=1000, bias=True, is_classifier=True, 
                                                learning_rate=lr, early_stopping=True, clip_max=5, max_attempts=100)
                    time_start = perf_counter()
                    clf_opt.fit(X_train, Y_train)
                    fit_time = perf_counter() - time_start
                    yhat_opt_test = clf_opt.predict(X_test)
                    test_accuracy = accuracy_score(Y_test, yhat_opt_test)
                    f1_score_opt = f1_score(Y_test, yhat_opt_test, average='weighted')
                    try: results['fit_times'][key].append(fit_time)
                    except: 
                        results['fit_times'][key] = []
                        results['fit_times'][key].append(fit_time)
                    try: results['accuracies'][key].append(test_accuracy)
                    except: 
                        results['accuracies'][key] = []
                        results['accuracies'][key].append(test_accuracy)
                    try: results['f1_scores'][key].append(f1_score_opt)
                    except: 
                        results['f1_scores'][key] = []
                        results['f1_scores'][key].append(f1_score_opt)

   
    # Plotting Accuracy by learning rate
    plt.figure(figsize=(12, 6))
    metric = 'accuracies'
    for algo in algorithms:
        lr_values = []
        metric_values = []
        for lr in learning_rates:
            key = f'{algo} {lr} lr'  # Use adjusted key format
            for k, value in results[metric].items():
                if key in k:
                    lr_values.append(lr)
                    metric_values.append(value)
        plt.scatter(lr_values, metric_values, label=algo, alpha=0.6)

    plt.title(f'{metric.capitalize()} by Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel(metric.capitalize())
    plt.xticks(lr_values)  # Ensure all learning rates are shown on the x-axis
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/validation_by_{metric}.png')


    # Plotting f1 by learning rate
    plt.figure(figsize=(12, 6))
    metric = 'f1_scores'
    for algo in algorithms:
        lr_values = []
        metric_values = []
        for lr in learning_rates:
            key = f'{algo} {lr} lr'  # Use adjusted key format
            for k, value in results[metric].items():
                if key in k:
                    lr_values.append(lr)
                    metric_values.append(value)
        plt.scatter(lr_values, metric_values, label=algo, alpha=0.6)

    plt.title(f'{metric.capitalize()} by Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel(metric.capitalize())
    plt.xticks(lr_values)  # Ensure all learning rates are shown on the x-axis
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/validation_by_{metric}.png')

    # Plotting fit time by learning rate
    plt.figure(figsize=(12, 6))
    metric = 'fit_times'
    for algo in algorithms:
        lr_values = []
        metric_values = []
        for lr in learning_rates:
            # Collect metrics for the specific algorithm and learning rate
            key = f'{algo} {lr} lr'  # Use adjusted key format
            if key in results[metric]:
                for value in results[metric][key]:  # Use all obtained values
                    lr_values.append(lr)
                    metric_values.append(value)
        # Plot each algorithm's results
        plt.scatter(lr_values, metric_values, label=algo, alpha=0.6)

    plt.title(f'{metric.capitalize()} by Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel(metric.capitalize())
    plt.xticks(lr_values)  # Ensure all learning rates are shown on the x-axis
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/validation_by_{metric}.png')


if __name__ == "__main__":
    run_nn_validation_experiment()
    