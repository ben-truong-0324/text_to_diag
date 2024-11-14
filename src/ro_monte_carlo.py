import numpy as np
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
from time import time

from config import *
from data_plots import *
import data_etl

import re
import time
import os
from datetime import datetime
import unittest
import inspect
# Helper function to create problems
def create_problem(problem_type, size):
    if problem_type == 'knapsack':
        weights = np.random.randint(1, 100, size)
        values = np.random.randint(1, 100, size)
        fitness = mlrose.Knapsack(weights, values, MAX_WEIGHT_PCT) #define eval func
        problem = mlrose.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True, max_val=2)

    elif problem_type == 'tsp':
        # coords = np.random.rand(size, 2) * 100 + .01
        # coords = np.array(coords*100) 
        # coords_list = coords.tolist() 
        # fitness = mlrose.TravellingSales(coords=coords_list)
        # problem = mlrose.TSPOpt(length=PROBLEM_SIZE_DEFAULT, fitness_fn=fitness, maximize=False)

        # coords = np.random.rand(size, 2) * 100 + .01
        # fitness = mlrose.TravellingSales(coords=coords) #eval func
        # problem = mlrose.TSPOpt(length=size, fitness_fn=fitness, maximize=False)

        coords = np.random.rand(PROBLEM_SIZE_DEFAULT, 2) 
        coords = np.array(coords*100) 
        coords_list = coords.tolist() 
        fitness = mlrose.TravellingSales(coords=coords_list)
        problem = mlrose.TSPOpt(length=size, fitness_fn=fitness, maximize=False)


    return problem

# Function to run algorithms and collect data
def run_algorithms(problem_type):
    problem = create_problem(problem_type, PROBLEM_SIZE_DEFAULT)
    algorithms = {
        'RHC': mlrose.random_hill_climb,
        'SA': mlrose.simulated_annealing,
        'GA': mlrose.genetic_alg,
        'MIMIC': mlrose.mimic
    }
    random_number = np.random.random()

    all_results = {name: [] for name in algorithms.keys()}  # Collect results for each algorithm

    for iteration in range(MONTE_CARLO_ITER):
        print(iteration)
        random_number = np.random.random()  # Generate a new random seed for each iteration
        for name, algo in algorithms.items():
            start_time = time.time()
            if name == 'RHC':
                _, fitness, fitness_curve = algo(problem, restarts=RESTARTS, max_attempts=MAX_ATTEMPTS, 
                                                  max_iters=MAX_ITERS, curve=True, random_state=random_number)
            elif name == 'SA':
                _, fitness, fitness_curve = algo(problem, schedule=mlrose.ExpDecay(), max_attempts=MAX_ATTEMPTS, 
                                                  max_iters=MAX_ITERS, curve=True, random_state=random_number)
            elif name == 'GA':
                _, fitness, fitness_curve = algo(problem, pop_size=POP_SIZE, mutation_prob=MUTATION_PROB, 
                                                  max_attempts=MAX_ATTEMPTS, max_iters=MAX_ITERS, curve=True, 
                                                  random_state=random_number)
            else:  # MIMIC
                _, fitness, fitness_curve = algo(problem, pop_size=POP_SIZE, keep_pct=KEEP_PCT, 
                                                  max_attempts=MAX_ATTEMPTS, max_iters=MAX_MIMIC_ITER, 
                                                  curve=True, random_state=random_number)
            elapsed_time = time.time() - start_time
            # Collect results for this iteration
            if 'tsp' in problem_type: all_results[name].append({'fitness': -fitness_curve[:, 0], 'time': elapsed_time})
            else: all_results[name].append({'fitness': fitness_curve[:, 0], 'time': elapsed_time})
    return all_results


def plot_monte_carlo_results(all_results, title, xlabel, ylabel,output_dir):    
    plt.figure(figsize=(12, 6))
    for algo, runs in all_results.items():
        for run in runs:
            # Each run might have a different number of iterations
            fitness = run['fitness']
            iterations = np.arange(len(fitness))  # Create x-axis for iterations
            plt.plot(iterations, fitness, color=ALGO_COLORS[algo], alpha=0.5, label=algo if algo not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/monte_{title}.png')

def plot_monte_carlo_time_results(all_results, title, xlabel, ylabel,output_dir):
    plt.figure(figsize=(12, 6))
    algorithms = list(all_results.keys())
    dot_positions = np.arange(len(algorithms))  # Position for each algorithm
    for i, algo in enumerate(algorithms):
        times = [run['time'] for run in all_results[algo]]
        plt.scatter(np.full_like(times, dot_positions[i]), times, label=algo)

    plt.xticks(dot_positions, algorithms)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.savefig(f'{output_dir}/monte_{title}.png')

from scipy import stats

def print_monte_carlo_time_stats(all_results,output_dir):
    stats_summary = {}
    
    # Open the file for writing
    with open(f'{output_dir}/monte_t_test.txt', 'w') as f:
        f.write("Average and Standard Deviation of Run Times:\n")
        f.write("="*40 + "\n")

        # Calculate averages and standard deviations
        for algo, runs in all_results.items():
            times = [run['time'] for run in runs]
            avg_time = np.mean(times)
            std_time = np.std(times)
            stats_summary[algo] = {'Average': avg_time, 'Std Dev': std_time}
            
            # Print and save to file
            output = f"{algo}: Average Time = {avg_time:.4f}s, Std Dev = {std_time:.4f}s"
            print(output)
            f.write(output + "\n")
        
        f.write("\nT-tests between Algorithms:\n")
        f.write("="*40 + "\n")
        
        # Perform t-tests
        algo_list = list(all_results.keys())
        for i in range(len(algo_list)):
            for j in range(i + 1, len(algo_list)):
                algo1 = algo_list[i]
                algo2 = algo_list[j]
                times1 = [run['time'] for run in all_results[algo1]]
                times2 = [run['time'] for run in all_results[algo2]]
                t_stat, p_val = stats.ttest_ind(times1, times2, equal_var=False)
                t_test_output = f"T-test between {algo1} and {algo2}: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}"
                
                # Print and save to file
                print(t_test_output)
                f.write(t_test_output + "\n")


def run_experiment(problem_type):

    fitness_by_algo = {}
    time_by_algo = {}
  
    results = run_algorithms(problem_type)

    # for algo, data in results.items():
        
    #     fitness_by_algo[algo] = data['fitness']
    #     if problem_type =='tsp': fitness_by_algo[algo] *= (-1) #flip TSP minima to show for "maxima"
    #     time_by_algo[algo] = data['time']

    output_dir = f'{OUTPUT_DIR_OPTIMIZE}/ver{OPT_DRAFT_VER}'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # plot_results(fitness_by_algo, f'{problem_type.upper()} - Fitness by Algo', 'Iteration', 'Fitness')
    # plot_time_results(time_by_algo, f'{problem_type.upper()} - Run Time by Algo', 'Algo', 'Time (s)')

    plot_monte_carlo_results(results, f'{problem_type.upper()} - Fitness by Algo', 'Iteration', 'Fitness',output_dir)
    plot_monte_carlo_time_results(results, f'{problem_type.upper()} - Run Time by Algo', 'Algo', 'Time (s)',output_dir)
    print_monte_carlo_time_stats(results,output_dir)

def run_ro_monte_carlo():
    run_experiment('knapsack')
    run_experiment('tsp')

if __name__ == "__main__":
    run_ro_monte_carlo()