

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier

import mlrose_hiive as mlrose
# import mlrose as mlrose

from config import *
import data_etl
from sklearn.metrics import confusion_matrix,precision_score, \
                        recall_score,classification_report, \
                        accuracy_score, f1_score

import re
import time
import os
from datetime import datetime
import unittest


def plot_fitness_curves(results, output_dir):
    # Extract algorithms for plotting
    algorithms = list(results.keys())
    
    # Create a plot for each algorithm's fitness curve
    plt.figure(figsize=(12, 8))  # Adjust size as necessary


    
    for algorithm in algorithms:
        fitness_curve = results[algorithm]['fitness_curve']
        if isinstance(fitness_curve, np.ndarray) and len(fitness_curve.shape) == 2:
            # If 2D, extract the first column
            fitness_curve = fitness_curve[:, 0]
        elif isinstance(fitness_curve, list) and isinstance(fitness_curve[0], (list, np.ndarray)):
            # If it's a list of lists (in case it's not an ndarray), convert to array and take the first column
            fitness_curve = np.array(fitness_curve)[:, 0]
        # fitness_curve = results[algorithm]['fitness_curve']  # Extract the fitness curve
        
        plt.plot(fitness_curve, label=f"{algorithm} Loss Curve")  # Plot the curve

    # Set plot title and labels
    plt.title("Loss Curves for Algorithms")
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Loss")
    
    # Add a legend
    plt.legend(loc="upper right")
    
    # Add grid for better visualization
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/baseline_fitness_curves.png')


def plot_wall_clock_time(results,output_dir):
    algorithms = list(results.keys())
    times = [result['training_time'] for result in results.values()]

    plt.bar(algorithms, times)
    plt.xlabel('Algorithm')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/baseline_training_times_comparison.png')
    plt.close()

def plot_results(results, output_dir):
    # Extract metrics for plotting
    algorithms = list(results.keys())
    accuracies = [results[algo]['accuracy'] for algo in algorithms]
    f1_scores = [results[algo]['f1'] for algo in algorithms]
    recalls = [results[algo]['recall'] for algo in algorithms]
    precisions = [results[algo]['precision'] for algo in algorithms]

    # Create a 2x2 figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid

    # Plot accuracy
    axs[0, 0].bar(algorithms, accuracies, color='skyblue')
    axs[0, 0].set_title('Model Accuracy by Algorithm')
    axs[0, 0].set_xlabel('Algorithm')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_ylim(0, 1)  # Accuracy ranges from 0 to 1
    axs[0, 0].grid(axis='y')

    # Plot F1 score
    axs[0, 1].bar(algorithms, f1_scores, color='green')
    axs[0, 1].set_title('F1 Score by Algorithm')
    axs[0, 1].set_xlabel('Algorithm')
    axs[0, 1].set_ylabel('F1 Score')
    axs[0, 1].set_ylim(0, 1)  # F1 score ranges from 0 to 1
    axs[0, 1].grid(axis='y')

    # Plot recall
    axs[1, 0].bar(algorithms, recalls, color='orange')
    axs[1, 0].set_title('Recall by Algorithm')
    axs[1, 0].set_xlabel('Algorithm')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].set_ylim(0, 1)  # Recall ranges from 0 to 1
    axs[1, 0].grid(axis='y')

    # Plot precision
    axs[1, 1].bar(algorithms, precisions, color='purple')
    axs[1, 1].set_title('Precision by Algorithm')
    axs[1, 1].set_xlabel('Algorithm')
    axs[1, 1].set_ylabel('Precision')
    axs[1, 1].set_ylim(0, 1)  # Precision ranges from 0 to 1
    axs[1, 1].grid(axis='y')

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{output_dir}/baseline_metrics.png')

def run_nn_baseline_experiment():
    output_dir = f'{OUTPUT_DIR_OPTIMIZE}/ver{OPT_DRAFT_VER}nn'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    X, y = data_etl.get_data(DATASET_SELECTION, 1, 0)  # dataset, do_scaling, do_pca
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=GT_ID)

    # Define the neural network structure
    hidden_nodes = [16, 8]
    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 
                        'gradient_descent', ]

    results = {}  # Using a dictionary to store results
    nn_models = []

    cm = confusion_matrix(y_test, y_test)

    # Train and evaluate models for each algorithm
    for algorithm in algorithms:
        start_time = time.time()
        nn_model = mlrose.NeuralNetwork(
            # hidden_nodes=HIDDEN_NODES, 
        activation='relu', algorithm=algorithm,
                                max_iters=1000, bias=True, is_classifier=True,
                                learning_rate=0.05, early_stopping=True, clip_max=5, curve=True,
                                random_state=GT_ID)
        end_time = time.time()

        training_time = end_time - start_time

        nn_model.fit(X_train, y_train)

        y_pred = nn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted') 
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        
        fitness_curve = nn_model.fitness_curve
       
        results[algorithm] = {
            'accuracy': accuracy,
            'f1': f1,
            'training_time': training_time,
            'recall':recall,
            'precision':precision,
            'fitness_curve': fitness_curve,
        }
        
    # Plot results
    plot_wall_clock_time(results,output_dir)
    plot_results(results,output_dir)
    plot_fitness_curves(results,output_dir)

if __name__ == "__main__":
    run_nn_baseline_experiment()
    