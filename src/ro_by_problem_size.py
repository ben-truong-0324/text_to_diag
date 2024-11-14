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
        # fitness = mlrose.TravellingSales(coords=coords) #eval func
        # problem = mlrose.TSPOpt(length=size, fitness_fn=fitness, maximize=False)
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
        print(f"running {name} in ro_by_prob_size")
        if EXP_DEBUG: 
            print(inspect.signature(algo))
            # help(algo)

        start_time = time.time()
        if name == 'RHC':
            _, fitness, fitness_curve = algo(problem, restarts=RESTARTS, max_attempts=MAX_ATTEMPTS, max_iters=MAX_ITERS, curve=True, 
            # random_state=random_number, 
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
    colors = ALGO_COLORS
    for algo, data in results.items():
        plt.plot(data['x'], data['y'], label=algo, color=colors[algo])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/psize_{title}.png')
    plt.show()

def run_experiment(problem_type):
    # Experiment 1: Fitness by problem size, runtime by problem size
    sizes = PROBLEM_SIZES

    fitness_by_size = {algo: {'x': sizes, 'y': []} for algo in RO_ALGORITHMS}
    time_by_size = {algo: {'x': sizes, 'y': []} for algo in RO_ALGORITHMS}
    for size in sizes:
        problem = create_problem(problem_type, size)
        print(f"runnnig for {problem_type} psize {size}")
        results = run_algorithms(problem)
        for algo, data in results.items():
            fitness_by_size[algo]['y'].append(data['fitness'][-1])
            time_by_size[algo]['y'].append(data['time'])
        if 'tsp' in problem_type: fitness_by_size[algo]['y'] *= (-1)

    plot_results(fitness_by_size, f'{problem_type.upper()} - Fitness by Problem Size', 'Problem Size', 'Fitness')
    plot_results(time_by_size, f'{problem_type.upper()} - Run Time by Problem Size', 'Problem Size', 'Time (s)')

def run_ro_by_problem_size():
    run_experiment('knapsack')
    run_experiment('tsp')

if __name__ == "__main__":
    run_ro_by_problem_size()