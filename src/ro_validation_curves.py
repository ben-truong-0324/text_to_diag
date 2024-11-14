import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
import time
from config import *
import os

def create_problem(problem_type):
    if problem_type == 'knapsack':
        weights = np.random.randint(1, 31, size=PROBLEM_SIZE_DEFAULT)  # Random weights between 1 and 30
        values = np.random.randint(1, 31, size=PROBLEM_SIZE_DEFAULT)    
        fitness = mlrose.Knapsack(weights, values, MAX_WEIGHT_PCT)
        problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

    elif problem_type == 'tsp':
        coords = np.random.rand(PROBLEM_SIZE_DEFAULT, 2) 
        coords = np.array(coords*100) 
        coords_list = coords.tolist() 
        fitness = mlrose.TravellingSales(coords=coords_list)
        problem = mlrose.TSPOpt(length=PROBLEM_SIZE_DEFAULT, fitness_fn=fitness, maximize=False)
    return problem

def run_rhc_val(initial_states, problem):
    fitness_rhc = []
    rhc_runtimes = []
    print("val for RHC")
    for init_state in initial_states:
        start_time = time.time()
        _, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(
            problem, max_attempts=MAX_ATTEMPTS, max_iters=MAX_ITERS, 
            curve=True, init_state=init_state
        )
        fitness_rhc.append(-fitness_curve_rhc[:, 0])  # Record fitness curve
        rhc_runtimes.append(time.time() - start_time)

    
    # Plot for Randomized Hill Climbing
    plt.figure(figsize=(8, 6))
    for idx, curve in enumerate(fitness_rhc):
        plt.plot(curve, label=f'Initial state {initial_states[idx]}')
    plt.title('Randomized Hill Climbing by Initial States')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.savefig(f'{output_dir}/val_rhc_{problem}.png')  # Save the plot as a PNG file
    plt.close()  # Close the figure


# Function to run algorithms and collect data
def run_experiment(problem_type):
    output_dir = f'{OUTPUT_DIR_OPTIMIZE}/ver{OPT_DRAFT_VER}'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    problem = create_problem(problem_type)
    num_hyperval=HYPERPARAM_VALUES_TO_VALIDATE
    # Initialize empty lists to store fitness values
    fitness_rhc = []
    fitness_sa = []
    fitness_ga = []
    fitness_mimic = []
    rhc_runtimes = []
    sa_runtimes = []
    ga_runtimes = []
    mimic_runtimes = []

    initial_temperatures = np.linspace(70, 150, HYPERPARAM_VALUES_TO_VALIDATE).tolist()  # for SA
    cooling_schedules = np.linspace(0.7, 0.99, HYPERPARAM_VALUES_TO_VALIDATE).tolist()  # for SA and MIMIC
    population_sizes = np.random.randint(5, 50, HYPERPARAM_VALUES_TO_VALIDATE).tolist()  # for GA and MIMIC
    mutation_rates = np.linspace(0.01, 0.5, HYPERPARAM_VALUES_TO_VALIDATE).tolist()  # for GA

    # np.random.randint(0, 2, size=PROBLEM_SIZE_DEFAULT).tolist()
    if 'knap' in problem_type: 
        initial_states = [np.random.randint(2, size=PROBLEM_SIZE_DEFAULT) for _ in range(PROBLEM_SIZE_DEFAULT)]  # Random initial states 
    else: 
        initial_states = [
            np.random.permutation(PROBLEM_SIZE_DEFAULT),
            np.random.permutation(PROBLEM_SIZE_DEFAULT),
            np.random.permutation(PROBLEM_SIZE_DEFAULT)
        ]

    # # 1. Randomized Hill Climbing with varying initial states
    

    print("val for SA")
    # Create dictionaries to hold fitness curves categorized by initial temperature and cooling schedule
    fitness_by_temp = {}
    fitness_by_cooling = {}

    # Iterate through initial temperatures and cooling schedules
    for initial_temp in initial_temperatures:
        for cooling_schedule in cooling_schedules:
            for iteration in range(MONTE_CARLO_ITER):
                random_number = np.random.random()
                # print(iteration)
                _, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(
                    problem, max_attempts=MAX_ATTEMPTS, max_iters=MAX_ITERS, random_state=random_number,
                    curve=True, schedule=mlrose.GeomDecay(init_temp=initial_temp, decay=cooling_schedule)
                )
                
                # Store fitness curves categorized by initial temperature
                if initial_temp not in fitness_by_temp:
                    fitness_by_temp[initial_temp] = []
                if 'tsp' in problem_type: fitness_by_temp[initial_temp].append(-fitness_curve_sa[:, 0])  
                else: fitness_by_temp[initial_temp].append(fitness_curve_sa[:, 0])  
                # Store fitness curves categorized by cooling schedule
                if cooling_schedule not in fitness_by_cooling:
                    fitness_by_cooling[cooling_schedule] = []
                if 'tsp' in problem_type: fitness_by_cooling[cooling_schedule].append(-fitness_curve_sa[:, 0])  
                else: fitness_by_cooling[cooling_schedule].append(fitness_curve_sa[:, 0])  

   
  
    colors = ['red', 'blue', 'green', 'black', 'orange']

    plt.figure(figsize=(8, 6))

    for i, (initial_temp, curves) in enumerate(fitness_by_temp.items()):
        color = colors[i % len(colors)]  # Cycle through the defined colors

        final_fitness_values = [] 

        
        for j, curve in enumerate(curves):
            final_fitness_values.append(curve[-1]) 
            if j == 0:
                plt.plot(curve, label=f'Initial Temp: {initial_temp}', color=color, alpha= .5)  # First curve gets the label
            else:
                plt.plot(curve, color=color, alpha= .5)  # Subsequent curves don't get a label

        avg_fitness = np.mean(final_fitness_values)  # Mean over curves (axis=0)
        std_fitness = np.std(final_fitness_values)   # Standard deviation over curves (axis=0)
        print(f'Initial Temp: {initial_temp}')
        print(f'Average Fitness: {avg_fitness}')
        print(f'Standard Deviation: {std_fitness}')
        print(f'std to avg: {std_fitness/avg_fitness}')
        print('---------------------------------------')
        


    plt.title('Simulated Annealing Fitness Curves by Initial Temperature')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/val_sa_by_temp_{problem_type}.png')
    plt.close()

    # Plotting by Cooling Schedule
    plt.figure(figsize=(8, 6))
    for i, (cooling_schedule, curves) in enumerate(fitness_by_cooling.items()):
        color = colors[i % len(colors)]  # Cycle through the defined colors

        final_fitness_values = [] 

        

        for j, curve in enumerate(curves):
            final_fitness_values.append(curve[-1]) 
            if j == 0:
                plt.plot(curve, label=f'Cooling Schedule: {cooling_schedule}', color=color,alpha = .6)  
            else:
                plt.plot(curve, color=color, alpha= .6)  # Subsequent curves don't get a label

        avg_fitness = np.mean(final_fitness_values, axis=0)  # Mean over curves (axis=0)
        std_fitness = np.std(final_fitness_values, axis=0)   # Standard deviation over curves (axis=0)

        print(f'Cooling Schedule: {cooling_schedule}')
        print(f'Average Fitness: {avg_fitness}')
        print(f'Standard Deviation: {std_fitness}')
        print(f'std to avg: {std_fitness/avg_fitness}')
        print('---------------------------------------')

    plt.title('Simulated Annealing Fitness Curves by Cooling Schedule')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/val_sa_by_cooling_{problem_type}.png')  # Save the plot as a PNG file
    plt.close()

    # # 3. Genetic Algorithm with varying population sizes, mutation rates, and crossover rates
    # print("val for GA")
    # for population_size in population_sizes:
    #     for mutation_rate in mutation_rates:
    #         start_time = time.time()
    #         _, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(
    #             problem, max_attempts=MAX_ATTEMPTS, max_iters=MAX_ITERS, 
    #             curve=True, pop_size=population_size, mutation_prob=mutation_rate, 
    #         )
    #         fitness_ga.append(-fitness_curve_ga[:, 0])
    #         ga_runtimes.append(time.time() - start_time)

    # 4. MIMIC with varying population sizes and cooling schedules
    # print("val for MIMIC")
    # for population_size in population_sizes:
    #     for cooling_schedule in cooling_schedules:
    #         start_time = time.time()
    #         _, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(
    #             problem, max_attempts=MAX_ATTEMPTS, max_iters=MAX_MIMIC_ITER, 
    #             curve=True, pop_size=population_size, keep_pct=cooling_schedule
    #         )
    #         if 'tsp' in problem_type: fitness_mimic.append(-fitness_curve_mimic[:, 0])
    #         else: fitness_mimic.append(fitness_curve_mimic[:, 0])
    #         mimic_runtimes.append(time.time() - start_time)


    # results_mimic = {pop_size: {'fitness_curve': [], 'runtimes': []} for pop_size in population_sizes}
    # print("val for MIMIC")
    # for population_size in population_sizes:
    #     for cooling_schedule in cooling_schedules:
    #         start_time = time.time()
    #         _, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(
    #             problem, max_attempts=MAX_ATTEMPTS, max_iters=MAX_MIMIC_ITER, 
    #             curve=True, pop_size=population_size, keep_pct=cooling_schedule
    #         )
    #         # Store the fitness curve for this population size
    #         if 'tsp' in problem_type:
    #             fitness_curve_mimic_values = -fitness_curve_mimic[:, 0]
    #         else:
    #             fitness_curve_mimic_values = fitness_curve_mimic[:, 0]
            
    #         # Store results grouped by population size
    #         results_mimic[population_size]['fitness_curve'].append(fitness_curve_mimic_values)
    #         results_mimic[population_size]['runtimes'].append(time.time() - start_time)


    

    

    # # Plot for Simulated Annealing
    # plt.figure(figsize=(8, 6))
    # for idx, (initial_temp, cooling_schedule) in enumerate([(temp, cool) for temp in initial_temperatures for cool in cooling_schedules]):
    #     plt.plot(fitness_sa[idx], label=f'Initial Temp {initial_temp}, Cool {cooling_schedule}')
    # plt.title('Simulated Annealing Fitness by Initial Temperatures and Cooling Schedule')
    # plt.xlabel('Iterations')
    # plt.ylabel('Fitness')
    # plt.savefig(f'{output_dir}/val_sa_{problem}.png')  # Save the plot as a PNG file
    # plt.close()  # Close the figure

    # Plot for Genetic Algorithm
    # plt.figure(figsize=(8, 6))
    # for idx, (population_size, mutation_rate) in enumerate([(size, rate) for size in population_sizes for rate in mutation_rates]):
    #     plt.plot(fitness_ga[idx], label=f'Population Size {population_size}, Mutation Rate {mutation_rate}')
    # plt.title('Genetic Algorithm Fitness by Population Size and Mutation Rate')
    # plt.xlabel('Iterations')
    # plt.ylabel('Fitness')
    # plt.savefig(f'{output_dir}/val_ga_{problem}.png')  # Save the plot as a PNG file
    # plt.close()  # Close the figure

  

    # plt.figure(figsize=(8, 6))

    # # Gather all fitness curves and determine the maximum length
    # all_fitness_curves = []
    # max_length = 0

    # # Collect fitness curves and find the maximum length
    # for population_size, data in results_mimic.items():
    #     for curve in data['fitness_curve']:
    #         all_fitness_curves.append(curve)
    #         max_length = max(max_length, len(curve))

    # # Prepare to align fitness curves by padding with NaNs
    # aligned_fitness_curves = []

    # # Pad each curve to the maximum length with NaNs
    # for curve in all_fitness_curves:
    #     padded_curve = np.pad(curve, (0, max_length - len(curve)), 'constant', constant_values=np.nan)
    #     aligned_fitness_curves.append(padded_curve)

    # # Plot each population size's results with a unique color
    # for population_size, data in results_mimic.items():
    #     for curve in data['fitness_curve']:
    #         # Pad the curve to the maximum length with NaNs
    #         padded_curve = np.pad(curve, (0, max_length - len(curve)), 'constant', constant_values=np.nan)
    #         plt.plot(padded_curve, label=f'Population Size: {population_size}', alpha=0.5)  # Use alpha for transparency

    # plt.title('MIMIC Fitness Curves by Population Size')
    # plt.xlabel('Iterations')
    # plt.ylabel('Fitness')
    # plt.legend()
    # plt.grid()
    # plt.savefig(f'{output_dir}/val_mimic_{problem_type}.png')  # Save the plot as a PNG file
    # plt.close()

    # Create a dot plot
    # all_runtimes = [rhc_runtimes, sa_runtimes, ga_runtimes, mimic_runtimes]
    # algorithms = ['RHC', 'SA', 'GA', 'MIMIC']
    # plt.figure(figsize=(10, 6))
    # for i, runtimes in enumerate(all_runtimes):
    #     plt.scatter(np.random.rand(len(runtimes)) + i, runtimes, label=f'{algorithms[i]}')
    # plt.title('Runtime Comparison')
    # plt.xlabel('Algorithm')
    # plt.ylabel('Runtime (seconds)')
    # plt.legend()
    # plt.savefig(f'{output_dir}/val_runtime_{problem}.png')


def run_ro_validation_curves():
    run_experiment('knapsack')
    run_experiment('tsp')

if __name__ == "__main__":
    np.random.seed(GT_ID)
    run_ro_validation_curves()
