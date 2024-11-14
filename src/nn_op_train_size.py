import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from time import perf_counter

import data_etl

from sklearn.model_selection import train_test_split
from config import *
from sklearn.neural_network import MLPClassifier
import mlrose_hiive as mlrose
import os



def plot_grouped_barchart(results, algorithms, metric_key, ylabel, output_dir, title):
    training_sizes = list(results[metric_key].keys())
    num_algorithms = len(algorithms)
    bar_width = 0.2  # Width of each bar
    index = np.arange(len(training_sizes))  # The x locations for the groups

    # Create figure
    plt.figure(figsize=(12, 8))

    # Loop over each algorithm to plot its results as a bar in each group
    for i, algo in enumerate(algorithms):
        # Get the metric values for the current algorithm across all training sizes
        metric_values = [results[metric_key][size][i] for size in training_sizes]
        
        # Calculate the position for each set of bars (shift by the bar width for grouping)
        plt.bar(index + i * bar_width, metric_values, bar_width, label=algo)

    # Set the x-axis labels to be the training sizes, and shift them so they are centered
    plt.xlabel('Training Size')
    plt.ylabel(ylabel)
    plt.ylim(.6, 1)  
    plt.title(title)
    plt.xticks(index + bar_width * (num_algorithms - 1) / 2, training_sizes)
    
    # Rotate x labels if needed
    plt.xticks(rotation=45)
    
    # Add legend and grid
    plt.legend()
    plt.grid(True, axis='y')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/train_size_{metric_key}_grouped.png')
    
    
def run_nn_by_train_size():
    output_dir = f'{OUTPUT_DIR_OPTIMIZE}/ver{OPT_DRAFT_VER}nn'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    # Set range for training set sizes
    train_sizes = np.linspace(0.7, 0.9, num=3)

    hidden_nodes_options = [
        # [2], [5], 
    [9], 
    #  [12], [15], [20]
    ]  # List of hidden nodes configurations
    learning_rates = [
        # 0.01, 
        0.05,
        # 0.1, 
        # 0.25,
        # 0.5,
        ]  # List of learning rates
    activation_functions = [
        'relu', 
    # 'sigmoid', 'tanh'
    ]  # List of activation functions

    results = {
        'fit_times': {},
        'accuracies': {},
        'f1_scores': {},
        'fitness_curve': {},
    }



    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent']


    # Loop over different training set sizes
    for train_size in train_sizes:
        # Load and split the dataset
        X_df, Y_df = data_etl.get_data("credit", 1, 0)
        X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, train_size=train_size, random_state=GT_ID)

        # Optimization Algorithms
        for algo in algorithms:
            print(train_size)
            print(algo)
            key = train_size
            # Initialize lists if key doesn't exist
            
            clf_opt = mlrose.NeuralNetwork(
                activation='relu', algorithm=algo, 
            #  hidden_nodes=nodes,
                max_iters=1000, bias=True, is_classifier=True, 
                learning_rate=.05, early_stopping=True, clip_max=5, max_attempts=100,curve=True)
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
            try: results['fitness_curve'][key].append(clf_opt.fitness_curve)
            except: 
                results['fitness_curve'][key] = []
                results['fitness_curve'][key].append(clf_opt.fitness_curve)

    # print(results)
    # Visualization
    plot_grouped_barchart(results, ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'], 
                      'accuracies', 'Accuracy', output_dir, 'Grouped Accuracy Bar Chart')
    plot_grouped_barchart(results, ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'], 
                      'f1_scores', 'F1 Score', output_dir, 'Grouped F1 Score Bar Chart')
    plot_grouped_barchart(results, 
                          ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'], 
                          'fit_times', 
                          'Fit Time (seconds)', 
                          output_dir, 
                          'Grouped Fit Time Bar Chart')

    plt.figure(figsize=(12, 8))  # Adjust size as necessary

    # Loop over the keys (training_set_size values)
    for training_size, fitness_curves in results['fitness_curve'].items():
        # Handle each fitness curve (since results['fitness_curve'][key] is a list)
        for fitness_curve in fitness_curves:
            # Check if fitness_curve is 2D or 1D
            if isinstance(fitness_curve, np.ndarray) and len(fitness_curve.shape) == 2:
                # If 2D, extract the first column
                fitness_curve = fitness_curve[:, 0]
            elif isinstance(fitness_curve, list) and isinstance(fitness_curve[0], (list, np.ndarray)):
                # If it's a list of lists (in case it's not an ndarray), convert to array and take the first column
                fitness_curve = np.array(fitness_curve)[:, 0]
            
            # Plot the fitness curve, using training_size in the label
            plt.plot(fitness_curve, label=f"Training size {training_size}")
    
    # Set plot title and labels
    plt.title("Fitness Curves by Training Set Size")
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Loss")
    
    # Add a legend
    plt.legend(loc="upper right")
    
    # Add grid for better visualization
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/train_size_fitness_curves.png')


    



if __name__ == "__main__":
    run_nn_by_train_size()