
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,precision_score,recall_score,classification_report

import re
import time
import os
from datetime import datetime
import unittest


def plot_fitness_curves(results, output_dir):
    plt.figure(figsize=(12, 8))
    
    # Define colors for each algorithm
    colors = {
        'random_hill_climb': 'blue',
        'simulated_annealing': 'orange',
        'genetic_alg': 'green',
        'gradient_descent': 'red'
    }

    # Plot each algorithm's fitness curves
    for algorithm, metrics in results.items():
        fitness_curves = metrics['fitness_curve']
        
        # Plot each fitness curve for the current algorithm
        for curve in fitness_curves:
            plt.plot(curve, color=colors[algorithm], alpha=0.6)  # Use alpha for transparency

    # Add labels and title
    plt.title('Fitness Curves by Algorithm', fontsize=16)
    plt.xlabel('Iterations/Epochs', fontsize=14)
    plt.ylabel('Normalized Fitness', fontsize=14)
    plt.grid()
    
    # Create a custom legend
    handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors.values()]
    plt.legend(handles, colors.keys(), title="Algorithms", loc='upper right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cv_fitness_curves.png')


def plot_wall_clock_time(results,output_dir):
    algorithms = list(results.keys())
    times = [result['training_time'] for result in results.values()]

    plt.bar(algorithms, times)
    plt.xlabel('Algorithm')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cv_training_times_comparison.png')
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
    plt.savefig(f'{output_dir}/cv_metrics.png')


def run_nn_cross_validation():
    output_dir = f'{OUTPUT_DIR_OPTIMIZE}/ver{OPT_DRAFT_VER}nn'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    X, y = data_etl.get_data(DATASET_SELECTION, 1, 0)  # dataset, do_scaling, do_pca
    # Define the neural network structure and algorithms
    hidden_nodes = [16, 8]
    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent']

    results = {}  # Using a dictionary to store results
    kf = KFold(n_splits=5, shuffle=True, random_state=GT_ID)  # 5-fold cross-validation

    for algorithm in algorithms:
        # Initialize lists to store metrics across folds
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        fitness_curves = []
        training_times = []

        # KFold cross-validation loop
        for fold, (train_index, test_index) in enumerate(kf.split(y)):
    
            print(f"Fold {fold+1}/5")
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Time the training process
            start_time = time.time()
            nn_model = mlrose.NeuralNetwork(
                hidden_nodes=hidden_nodes, activation='relu', algorithm=algorithm,
                max_iters=1000, bias=True, is_classifier=True, learning_rate=0.05,
                early_stopping=True, clip_max=5, curve=True, random_state=GT_ID
            )
            nn_model.fit(X_train, y_train)
            end_time = time.time()

            training_time = end_time - start_time
            y_pred = nn_model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Handle fitness curve
            fitness_curve = nn_model.fitness_curve
            if len(fitness_curve.shape) == 2:  # If 2D, take the first column
                fitness_curve = fitness_curve[:, 0]
            
            # Append metrics and fitness curve
            accuracies.append(accuracy)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            fitness_curves.append(fitness_curve)
            training_times.append(training_time)

        # After all folds, calculate the mean of the metrics across folds
        results[algorithm] = {
            'accuracy': np.mean(accuracies),
            'f1': np.mean(f1_scores),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'training_time': np.mean(training_times),
            'fitness_curve': fitness_curves  # Average fitness curve over folds
        }

    # Plot the results

    plot_wall_clock_time(results, output_dir)
    plot_results(results, output_dir)
    plot_fitness_curves(results, output_dir)

if __name__ == "__main__":
   
    run_nn_cross_validation()
    