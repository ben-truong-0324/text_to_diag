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
def run_algorithms(problem):
    algorithms = {
        'RHC': mlrose.random_hill_climb,
        'SA': mlrose.simulated_annealing,
        'GA': mlrose.genetic_alg,
        'MIMIC': mlrose.mimic
    }
    random_number = np.random.random()

    results = {}
    for name, algo in algorithms.items():
        print(f"running {name} in ro_baseline")
        if EXP_DEBUG: 
            print(inspect.signature(algo))
            # help(algo)

        start_time = time.time()
        if name == 'RHC':
            _, fitness, fitness_curve = algo(problem, restarts=RESTARTS, max_attempts=MAX_ATTEMPTS, max_iters=MAX_ITERS, curve=True, random_state=random_number, 
            )
        elif name == 'SA':
            _, fitness, fitness_curve = algo(problem, schedule=mlrose.ExpDecay(), max_attempts=MAX_ATTEMPTS, max_iters=MAX_ITERS, curve=True, random_state=random_number, 
            )
        elif name == 'GA':
            _, fitness, fitness_curve = algo(problem, pop_size=POP_SIZE, mutation_prob=MUTATION_PROB, max_attempts=MAX_ATTEMPTS, max_iters=MAX_ITERS, curve=True, random_state=random_number,
             )
        else:  # MIMIC
            _, fitness, fitness_curve = algo(problem, pop_size=POP_SIZE, keep_pct=KEEP_PCT, max_attempts=MAX_ATTEMPTS, max_iters=MAX_MIMIC_ITER, curve=True, random_state=random_number,
             )
        results[name] = {'fitness': fitness_curve[:, 0], 'time': time.time() - start_time}
    return results

# Function to plot results
def plot_results(results, title, xlabel, ylabel):
    output_dir = f'{OUTPUT_DIR_OPTIMIZE}/ver{OPT_DRAFT_VER}'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    plt.figure(figsize=(10, 6))
    for algo, data in results.items():
        plt.plot(range(len(data)), data, label=algo, color=ALGO_COLORS[algo])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/baseline_{title}.png')
    plt.show()

# Function to plot results
def plot_time_results(results, title, xlabel, ylabel):
    output_dir = f'{OUTPUT_DIR_OPTIMIZE}/ver{OPT_DRAFT_VER}'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    plt.figure(figsize=(10, 6))

    bars = plt.bar(results.keys(), results.values(), color=[ALGO_COLORS[algo] for algo in results.keys()])
    # Adding value annotations on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/baseline_{title}.png')
    plt.show()

def run_experiment(problem_type):

    fitness_by_algo = {}
    time_by_algo = {}
  
    problem = create_problem(problem_type, PROBLEM_SIZE_DEFAULT)
    results = run_algorithms(problem)

    for algo, data in results.items():
        
        fitness_by_algo[algo] = data['fitness']
        if problem_type =='tsp': fitness_by_algo[algo] *= (-1) #flip TSP minima to show for "maxima"

        time_by_algo[algo] = data['time']

    plot_results(fitness_by_algo, f'{problem_type.upper()} - Fitness by Algo', 'Iteration', 'Fitness')
    plot_time_results(time_by_algo, f'{problem_type.upper()} - Run Time by Algo', 'Algo', 'Time (s)')

def run_ro_baseline_experiment():
    run_experiment('knapsack')
    run_experiment('tsp')

if __name__ == "__main__":
    run_ro_baseline_experiment()