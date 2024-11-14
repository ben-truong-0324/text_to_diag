import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
import numpy as np
import os 
from sklearn.metrics import silhouette_score
import pickle

from config import *

def plot_fitness_iterations(nn_models, algorithm_names):
    plt.figure(figsize=(12, 6))
    for model, name in zip(nn_models, algorithm_names):
        if hasattr(model, 'fitness_curve'):
            plt.plot(model.fitness_curve, label=name)
    
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Score')
    plt.title('Fitness vs. Iteration')
    plt.legend()
    plt.grid()
    plt.savefig(f"{OUTPUT_DIR}/fitness_vs_iteration.png")
    plt.close()




def plot_training_loss(train_losses, algorithm, activation_name):
    plt.plot(train_losses, label=f'{activation_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for {algorithm}')
    plt.legend()
    plt.grid()
    plt.savefig(f'{OUTPUT_DIR}/training_loss_{algorithm}.png')
    plt.close()

def plot_wall_clock_time(results):
    algorithms = list(results.keys())
    times = [result['time'] for result in results.values()]

    plt.bar(algorithms, times)
    plt.xlabel('Algorithm')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('{OUTPUT_DIR}/training_times_comparison.png')
    plt.close()


def graph_class_imbalance(y, outpath):
    start_time = time.time()
    
    # Calculate the percentage of each class using np.unique
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_percentages = (class_counts / y.shape[0]) 

    # Create a bar plot for the percentage of each class
    plt.figure(figsize=(6, 4))
    sns.barplot(x=unique_classes, y=class_percentages, palette='coolwarm', dodge=False)

    plt.title('Distribution of Labels (%)')
    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    
    # Add percentage labels on top of the bars
    for i, value in enumerate(class_percentages):
        plt.text(i, value + 1, f'{value:.2f}%', ha='center', va='bottom')
    
    end_time = time.time()
    print(("Time to graph class imbalance: " + str(end_time - start_time) + "s"))
    
    # Adding the time to the graph
    plt.text(-0.5, 1, f'Time to graph: {(end_time - start_time):.4f}s', fontsize=12, color='black', 
             ha='left', va='top')
    
    plt.tight_layout()  # Adjust the layout
    plt.savefig(outpath)
    plt.close() 
	

def graph_feature_correlation(X_df, y, outpath):
    start_time = time.time()

    # Convert y to a DataFrame and combine with X_df
    y_df = pd.Series(y, name='class')  # Convert y to a pandas Series
    combined_df = pd.concat([X_df, y_df], axis=1)  # Combine X_df and y_df

    # Plot correlations with subplots
    n_cols = 2
    n_rows = (len(X_df.columns) + n_cols - 1) // n_cols 
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 16))
    axes = axes.flatten()

    for i, feature in enumerate(X_df.columns):
        # Plotting the boxplot using the combined DataFrame
        sns.boxplot(x='class', y=feature, data=combined_df, ax=axes[i])
        axes[i].set_title(f'Feature: {feature} vs class')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel(feature)

    end_time = time.time()
    print(("Time to graph feature correlation: " + str(end_time - start_time) + "s"))

    # Adding graph time to the plot
    plt.text(0, -0.5, f'Time to graph: {(end_time - start_time):.4f}s', fontsize=12, color='black', 
             ha='left', va='top')

    plt.tight_layout()  # Adjust the layout
    plt.savefig(outpath)
    plt.close()
	
def graph_feature_histogram(X_df, outpath):
    start_time = time.time()
    # Plot correlations with subplots
    n_cols = 2
    n_rows = (len(X_df.columns) + n_cols - 1) // n_cols 
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 16))
    axes = axes.flatten()
    for i, feature in enumerate(X_df.columns):
        feature_kurtosis = X_df[feature].kurt()
        sns.histplot(X_df[feature], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f'Histogram distribution of {feature} (Kurtosis: {feature_kurtosis:.2f})')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
    end_time = time.time()
    print(("Time to graph feature histogram: " + str(end_time - start_time) + "s"))
    plt.text(0, -0.5, f'Time to graph: {(end_time - start_time):.4f}s', fontsize=12, color='black', 
            ha='left', va='top')
    plt.tight_layout()  # Adjust the layout
    plt.savefig(outpath)
    plt.close()
	

def graph_feature_heatmap(X_df, y, outpath):
    start_time = time.time()
    # Convert y to a DataFrame and combine with X_df
    y_df = pd.Series(y, name='Target')  # Convert y to a pandas Series
    combined_df = pd.concat([X_df, y_df], axis=1)  # Combine X_df and y_df
    
    # Calculate the correlation matrix
    corr = combined_df.corr()

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, 
    # annot=True, 
    cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Heatmap of Features and Target')
    
    end_time = time.time()
    print(("Time to graph feature heatmap: " + str(end_time - start_time) + "s"))
    
    # Adding graph time to the plot
    plt.text(0, -0.5, f'Time to graph: {(end_time - start_time):.4f}s', fontsize=12, color='black', 
             ha='left', va='top')

    plt.tight_layout()  # Adjust the layout
    plt.savefig(outpath)
    plt.close()


def graph_feature_boxplot(X_df, outpath):
	# Select the necessary features and labelIndex columns
	# Plot correlations with subplots
	start_time = time.time()
	n_cols = 2
	n_rows = (len(X_df.columns) + n_cols - 1) // n_cols 
	fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 16))
	axes = axes.flatten()
	for i, feature in enumerate(X_df.columns):
		sns.boxplot(x=X_df[feature], ax=axes[i])
		axes[i].set_title(f'Boxplot of {feature}')
		axes[i].set_xlabel(feature)
	end_time = time.time()
	print(("Time to graph feature boxplot: " + str(end_time - start_time) + "s"))
	plt.text(0, -0.5, f'Time to graph: {(end_time - start_time):.4f}s', fontsize=12, color='black', 
			ha='left', va='top')
	plt.tight_layout()  # Adjust the layout
	plt.savefig(outpath)
	plt.close()
	
def graph_feature_violin(X_df, y, outpath):
    start_time = time.time()
    # Convert y (numpy array) to pandas Series for easier plotting
    y_series = pd.Series(y, name='Target')

    # Combine X_df and y_series into a single DataFrame for plotting
    plot_df = pd.concat([X_df, y_series], axis=1)

    # Plot correlations with subplots
    n_cols = 2
    n_rows = (len(X_df.columns) + n_cols - 1) // n_cols 
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 16))
    axes = axes.flatten()

    for i, feature in enumerate(X_df.columns):
        feature_kurtosis = X_df[feature].kurt()
        # Use 'Target' as the categorical variable for x
        sns.violinplot(x='Target', y=feature, data=plot_df, ax=axes[i])
        axes[i].set_title(f'Violin plot of {feature} (kurtosis: {feature_kurtosis:.2f})')
        axes[i].set_xlabel("Target")
        axes[i].set_ylabel(feature)

    end_time = time.time()
    print(("Time to graph feature violin with kurt: " + str(end_time - start_time) + "s"))

    # Adding graph time to the plot
    plt.text(0, -0.5, f'Time to graph: {(end_time - start_time):.4f}s', fontsize=12, color='black', 
             ha='left', va='top')

    plt.tight_layout()  # Adjust the layout
    plt.savefig(outpath)
    plt.close()
	

def graph_feature_cdf(X_df, outpath):
	start_time = time.time()
	# Plot correlations with subplots
	n_cols = 2
	n_rows = (len(X_df.columns) + n_cols - 1) // n_cols 
	fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 16))
	axes = axes.flatten()
	for i, feature in enumerate(X_df.columns):
		sns.ecdfplot(data=X_df, x=feature, ax=axes[i])
		axes[i].set_title(f'Cumulative Distribution Function (CDF) of {feature}')
		axes[i].set_xlabel(feature)
	end_time = time.time()
	print(("Time to graph feature cdf: " + str(end_time - start_time) + "s"))
	plt.text(0, -0.5, f'Time to graph: {(end_time - start_time):.4f}s', fontsize=12, color='black', 
			ha='left', va='top')
	plt.tight_layout()  # Adjust the layout
	plt.savefig(outpath)
	plt.close()
	
def graph_cluster_runtime(results, cluster_algo, outpath):
    if not os.path.exists(outpath):
        # Extract data for plotting
        n_clusters = list(results.keys())
        runtimes = [results[n]['runtime'] for n in n_clusters]

        # Plot runtime vs. number of clusters
        plt.figure(figsize=(10, 6))
        plt.plot(n_clusters, runtimes, marker='o', color='b')
        plt.title(f'{cluster_algo} Runtime vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Runtime (seconds)')
        plt.grid(True)
        plt.savefig(outpath)
        plt.close()


def graph_cluster_silhouette(results, X, cluster_algo, outpath):
    if not os.path.exists(outpath):
        print("Starting silhouette score calculation...")
        n_clusters = list(results.keys())
        silhouette_scores = []

        # Start timing for the entire section
        section_start_time = time.time()

        # Loop over each cluster and calculate the silhouette score
        for idx, n in enumerate(n_clusters):
            print(f"Processing cluster {n} ({idx + 1}/{len(n_clusters)})...")

            # Start timing for this iteration
            iter_start_time = time.time()

            # Calculate the silhouette score
            labels = results[n]['labels']
            score = silhouette_score(X, labels)  # Assuming X is your feature data
            silhouette_scores.append(score)

            # Calculate and print the time taken for this iteration
            iter_end_time = time.time()
            iter_duration = iter_end_time - iter_start_time
            print(f"Cluster {n} processed in {iter_duration:.2f} seconds.")

        # Calculate the total time taken for this section
        section_end_time = time.time()
        total_duration = section_end_time - section_start_time
        print(f"Silhouette scores calculated for all clusters in {total_duration:.2f} seconds.")


        # Plot Silhouette Score vs. Number of Clusters
        plt.figure(figsize=(10, 6))
        plt.plot(n_clusters, silhouette_scores, marker='o', color='g')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        print("done with plotting")
        plt.grid(True)
        plt.savefig(outpath)
        plt.close()


def graph_cluster_count_per(results, cluster_algo, n_clusters_to_plot, outpath):
    if not os.path.exists(outpath):
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # Flatten the 2x2 grid to make indexing easier

        for i, n_clusters in enumerate(n_clusters_to_plot):
            labels = results[n_clusters]['labels']
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Plot cluster size distribution for each cluster number
            axes[i].bar(unique_labels, counts, color='c')
            axes[i].set_title(f'Cluster Size Distribution for {n_clusters} Clusters {cluster_algo}')
            axes[i].set_xlabel('Cluster Label')
            axes[i].set_ylabel('Number of Samples')
            axes[i].set_xticks(unique_labels)
            axes[i].grid(True)

        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()


def graph_cluster_count_per_for_all(results, cluster_algo, outpath):
    if not os.path.exists(outpath):
        plt.figure(figsize=(12, 8))

        for n_clusters in results.keys():
            labels = results[n_clusters]['labels']
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Plot the line for this n_clusters result
            plt.plot(unique_labels, counts, marker='o', label=f'{n_clusters} Clusters')

        # Title and labels
        plt.title(f'Cluster Size Distribution Across Different Cluster Numbers for {cluster_algo}')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Samples')
        plt.grid(True)
        plt.legend(title='Number of Clusters')
        plt.xticks(np.arange(min(unique_labels), max(unique_labels) + 1, 1))  # Ensure integer ticks for cluster labels

        # Save the plot
        plt.savefig(outpath)
        plt.close()

def plot_elbow_method(results, X, cluster_algo, outpath):
    if not os.path.exists(outpath):
        n_clusters_list = list(results.keys())
        inertia_values = []
        
        for n_clusters in n_clusters_list:
            labels = results[n_clusters]['labels']
            inertia = 0
            for label in np.unique(labels):
                cluster_points = X[labels == label].to_numpy() 
                centroid = cluster_points.mean(axis=0)  # Calculate centroid of the cluster
                # Sum of squared distances from points to the centroid
                inertia += np.sum((cluster_points - centroid) ** 2)
                # print(type(inertia))
            inertia_values.append(inertia)
            

        # Plot the Elbow Graph
        plt.figure(figsize=(10, 6))
        plt.plot(n_clusters_list, inertia_values, marker='o', linestyle='-', color='b')
        plt.title(f'Elbow Method for {cluster_algo}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia (WCSS)')
        plt.grid(True)
        plt.savefig(outpath)
        plt.close()


def graph_silhouette_and_runtime(results, X, cluster_algo, outpath):
    if not os.path.exists(outpath):
        n_clusters_list = list(results.keys())
        silhouette_scores = []
        runtimes = []

        for n_clusters in n_clusters_list:
            labels = results[n_clusters]['labels']
            runtime = results[n_clusters]['runtime']
            score = silhouette_score(X, labels)  # Assuming X is your feature data
            silhouette_scores.append(score)
            runtimes.append(runtime)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot silhouette scores on left y-axis
        ax1.plot(n_clusters_list, silhouette_scores, marker='o', color='g', label='Silhouette Score')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.grid(True)

        # Create a second y-axis for runtime
        ax2 = ax1.twinx()
        ax2.plot(n_clusters_list, runtimes, marker='o', color='b', label='Runtime')
        ax2.set_ylabel('Runtime (s)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        # Add titles and legends
        plt.title(f'Silhouette Score and Runtime vs. Number of Clusters ({cluster_algo})')
        fig.tight_layout()
        
        # Save the figure
        plt.savefig(outpath)
        plt.close()

def make_cluster_graphs(cluster_results,X,outpath, cluster_algo,tag):
    # graph_cluster_silhouette(cluster_results, X, cluster_algo,
    #             f'{outpath}/{tag}_{cluster_algo}_silhouette_by_k_cluster.png')
    graph_cluster_count_per_for_all(cluster_results, cluster_algo, 
                f'{outpath}/{tag}_{cluster_algo}_cluster_count_all.png')
    plot_elbow_method(cluster_results, X, cluster_algo, 
                f'{outpath}/{tag}_{cluster_algo}_elbow.png')
    # graph_silhouette_and_runtime(cluster_results, X, cluster_algo, 
    #             f'{outpath}/{tag}_{cluster_algo}_silhouette_vs_runtime.png')
    

def plot_cluster_usefulness_by_nn(nn_results, outpath_accuracy, outpath_f1):
    if not os.path.exists(outpath_accuracy) or os.path.exists(outpath_f1) :
        n_clusters = list(nn_results.keys())
        
        # Prepare data for plotting
        accuracies = []
        f1_scores = []
        runtimes = []

        # Extract baseline results
        baseline_mc_results = np.array(nn_results["baseline"]['mc_results'])
        baseline_accuracy = np.mean(baseline_mc_results[:, 0])
        baseline_f1_score = np.mean(baseline_mc_results[:, 1])
        baseline_runtime = np.mean(baseline_mc_results[:, 2])
        n_clusters.remove("baseline")

        for n in n_clusters:
            mc_results = np.array(nn_results[n]['mc_results'])  
            #refence func collect_cluster_usefulness_via_nn_wrapping()
            accuracies.append(np.mean(mc_results[:, 0]))  # Mean accuracy
            f1_scores.append(np.mean(mc_results[:, 1]))     # Mean F1 score
            runtimes.append(np.mean(mc_results[:, 2]))       # Mean runtime

        # Plot Accuracy and Runtime
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot accuracy
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Accuracy', color='tab:blue')
        ax1.plot(n_clusters, accuracies, marker='o', color='tab:blue', label='Mean Accuracy')
        # Baseline line
        ax1.axhline(y=baseline_accuracy, color='tab:orange', linestyle='--', linewidth=2, label='Baseline without Cluster Label')
        ax1.text(0, baseline_accuracy + 0.001, ' NN Baseline without Cluster Label', color='tab:orange', fontsize=10, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        # Create a second y-axis for runtime
        ax2 = ax1.twinx()  
        ax2.set_ylabel('Runtime (seconds)', color='black')  
        ax2.plot(n_clusters, runtimes, marker='x', color='black', label='Mean Runtime')
        ax2.tick_params(axis='y', labelcolor='black')
        # Add grid and title
        ax1.grid()
        plt.title('Mean Accuracy and Runtime by Number of Clusters')
        fig.tight_layout()  # Adjust layout to make room for the second y-axis
        plt.savefig(f'{outpath_accuracy}')
        plt.close()

        # Plot F1 Score and Runtime
        fig, ax3 = plt.subplots(figsize=(12, 6))
        # Plot F1 score
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('F1 Score', color='tab:green')
        # Baseline line
        ax3.axhline(y=baseline_f1_score, color='tab:orange', linestyle='--', linewidth=2, label='Baseline without Cluster Label')
        ax3.text(0, baseline_f1_score + 0.001, ' NN Baseline without Cluster Label', color='tab:orange', fontsize=10, fontweight='bold')
        ax3.plot(n_clusters, f1_scores, marker='o', color='tab:green', label='Mean F1 Score')
        ax3.tick_params(axis='y', labelcolor='tab:green')
        # Create a second y-axis for runtime
        ax4 = ax3.twinx()  
        ax4.set_ylabel('Runtime (seconds)', color='black')  
        ax4.plot(n_clusters, runtimes, marker='x', color='black', label='Mean Runtime')
        ax4.tick_params(axis='y', labelcolor='black')
        # Add grid and title
        ax3.grid()
        plt.title('Mean F1 Score and Runtime by Number of Clusters')
        fig.tight_layout()  # Adjust layout to make room for the second y-axis
        plt.savefig(f'{outpath_f1}')
        plt.close()

def plot_cluster_usefulness_by_nn_banded_mean(nn_results, outpath):
    if not os.path.exists(outpath):
        # Extract baseline results
        n_clusters = list(nn_results.keys())

        baseline_mc_results = np.array(nn_results["baseline"]['mc_results'])
        baseline_accuracy = np.mean(baseline_mc_results[:, 0])
        baseline_f1_score = np.mean(baseline_mc_results[:, 1])
        baseline_runtime = np.mean(baseline_mc_results[:, 2])

        # Remove "baseline" from n_clusters if present
        if "baseline" in n_clusters:
            n_clusters.remove("baseline")

        # Prepare lists for cluster-based results
        accuracies = []
        f1_scores = []
        runtimes = []
        accuracy_stds = []
        f1_score_stds = []
        runtime_stds = []


        # Collect results for each number of clusters
        for n in n_clusters:
            mc_results = np.array(nn_results[n]['mc_results'])
            accuracies.append(np.mean(mc_results[:, 0]))          # Mean accuracy
            accuracy_stds.append(np.std(mc_results[:, 0]))        # Std deviation of accuracy
            f1_scores.append(np.mean(mc_results[:, 1]))           # Mean F1 score
            f1_score_stds.append(np.std(mc_results[:, 1]))
            runtimes.append(np.mean(mc_results[:, 2]))            # Mean runtime
            runtime_stds.append(np.std(mc_results[:, 2])) 
        # Convert n_clusters to a numeric list if needed
        n_clusters_numeric = list(map(int, n_clusters))

        # Set up the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # First subplot: Accuracy and Runtime with dual y-axis
        color_accuracy = 'tab:blue'
        color_runtime = 'black'
        
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Accuracy', color=color_accuracy)
        ax1.plot(n_clusters_numeric, accuracies, color=color_accuracy, label='Accuracy')
        ax1.fill_between(n_clusters_numeric, 
                     np.array(accuracies) - np.array(accuracy_stds),
                     np.array(accuracies) + np.array(accuracy_stds),
                     color=color_accuracy, alpha=0.2, label='Accuracy ± 1 Std')
    
        ax1.axhline(baseline_accuracy, color=color_accuracy, linestyle='--', label='Baseline Accuracy')
        ax1.tick_params(axis='y', labelcolor=color_accuracy)

        ax1_runtime = ax1.twinx()
        ax1_runtime.set_ylabel('Runtime (s)', color=color_runtime)
        ax1_runtime.plot(n_clusters_numeric, runtimes, color=color_runtime, label='Runtime')
        ax1_runtime.fill_between(n_clusters_numeric, 
                             np.array(runtimes) - np.array(runtime_stds),
                             np.array(runtimes) + np.array(runtime_stds),
                             color=color_runtime, alpha=0.2, label='Runtime ± 1 Std')
    
        ax1_runtime.axhline(baseline_runtime, color=color_runtime, linestyle='--', label='Baseline Runtime')
        ax1_runtime.tick_params(axis='y', labelcolor=color_runtime)

        # Second subplot: F1 Score and Runtime with dual y-axis
        color_f1 = 'tab:green'
        
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('F1 Score', color=color_f1)
        ax2.plot(n_clusters_numeric, f1_scores, color=color_f1, label='F1 Score')
        ax2.fill_between(n_clusters_numeric, 
                     np.array(f1_scores) - np.array(f1_score_stds),
                     np.array(f1_scores) + np.array(f1_score_stds),
                     color=color_accuracy, alpha=0.2, label='F1 Score ± 1 Std')
        ax2.axhline(baseline_f1_score, color=color_f1, linestyle='--', label='Baseline F1 Score')
        ax2.tick_params(axis='y', labelcolor=color_f1)

        ax2_runtime = ax2.twinx()
        ax2_runtime.set_ylabel('Runtime (s)', color=color_runtime)
        ax2_runtime.plot(n_clusters_numeric, runtimes, color=color_runtime, label='Runtime')
        ax2_runtime.fill_between(n_clusters_numeric, 
                             np.array(runtimes) - np.array(runtime_stds),
                             np.array(runtimes) + np.array(runtime_stds),
                             color=color_runtime, alpha=0.2, label='Runtime ± 1 Std')
    
        ax2_runtime.axhline(baseline_runtime, color=color_runtime, linestyle='--', label='Baseline Runtime')
        ax2_runtime.tick_params(axis='y', labelcolor=color_runtime)

        # Add titles and layout adjustment
        fig.suptitle('Cluster Usefulness: Accuracy, F1, and Runtime with Baseline Reference')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the plot
        plt.savefig(outpath)
        print("saved to ",outpath)
        plt.close()



import numpy as np
import matplotlib.pyplot as plt

def plot_dreduced_usefulness_by_nn_acc_f1(nn_results, outpath):
    if not os.path.exists(outpath):  
        # print(nn_results)
        # Extract baseline results
        baseline_mc_results = np.array(nn_results["baseline"]['mc_results'])
        baseline_accuracy = np.mean(baseline_mc_results[:, 0])
        baseline_f1_score = np.mean(baseline_mc_results[:, 1])

        # Remove baseline from dreduced_types to avoid duplicates in plotting
        dreduced_types = list(nn_results.keys())
        dreduced_types.remove("baseline")

        # Initialize a dictionary to store results for each method
        method_results = {}

        # Collect results from nn_results
        for d in dreduced_types:
            method, k = d.split("_")
            mc_results = np.array(nn_results[d]['mc_results'])

            # Calculate means and standard deviations
            if method not in method_results:
                method_results[method] = {
                    'k': [],
                    'accuracies': [],
                    'accuracy_stds': [],
                    'f1_scores': [],
                    'f1_score_stds': []
                }

            method_results[method]['k'].append(int(k))
            method_results[method]['accuracies'].append(np.mean(mc_results[:, 0]))          # Mean accuracy
            method_results[method]['accuracy_stds'].append(np.std(mc_results[:, 0]))        # Std deviation of accuracy
            method_results[method]['f1_scores'].append(np.mean(mc_results[:, 1]))           # Mean F1 score
            method_results[method]['f1_score_stds'].append(np.std(mc_results[:, 1]))        # Std deviation of F1 score

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # First subplot: Accuracy
        for method, results in method_results.items():
            ax1.plot(results['k'], results['accuracies'], label=method, marker='o')
            ax1.fill_between(results['k'],
                            np.array(results['accuracies']) - np.array(results['accuracy_stds']),
                            np.array(results['accuracies']) + np.array(results['accuracy_stds']),
                            alpha=0.2)

        ax1.set_xlabel('k Dimension')
        ax1.set_ylabel('Accuracy')
        ax1.axhline(baseline_accuracy, color='tab:blue', linestyle='--', label='Baseline Accuracy')
        ax1.set_title('Accuracy vs. k Dimension')
        ax1.legend(loc='upper left')

        # Second subplot: F1 Score
        for method, results in method_results.items():
            ax2.plot(results['k'], results['f1_scores'], label=method, marker='o')
            ax2.fill_between(results['k'],
                            np.array(results['f1_scores']) - np.array(results['f1_score_stds']),
                            np.array(results['f1_scores']) + np.array(results['f1_score_stds']),
                            alpha=0.2)

        ax2.set_xlabel('k Dimension')
        ax2.set_ylabel('F1 Score')
        ax2.axhline(baseline_f1_score, color='tab:orange', linestyle='--', label='Baseline F1 Score')
        ax2.set_title('F1 Score vs. k Dimension')
        ax2.legend(loc='upper left')

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(outpath)
        print("saved to ",outpath)
        plt.close()

def plot_dreduced_usefulness_by_nn_banded_mean(nn_results, outpath):
    if not os.path.exists(outpath):
        print(nn_results)
        # Extract baseline results
        baseline_mc_results = np.array(nn_results["baseline"]['mc_results'])
        baseline_accuracy = np.mean(baseline_mc_results[:, 0])
        baseline_f1_score = np.mean(baseline_mc_results[:, 1])
        baseline_runtime = np.mean(baseline_mc_results[:, 2])

        # Remove baseline from dreduced_types to avoid duplicates in plotting
        dreduced_types = list(nn_results.keys())
        dreduced_types.remove("baseline")

        # Initialize a dictionary to store results for each method
        method_results = {}

        # Collect results from nn_results
        for d in dreduced_types:
            print(d)
            method, k = d.split("_")
            mc_results = np.array(nn_results[d]['mc_results'])
            print(mc_results)

            # Calculate means and standard deviations
            if method not in method_results:
                method_results[method] = {
                    'k': [],
                    'accuracies': [],
                    'accuracy_stds': [],
                    'f1_scores': [],
                    'f1_score_stds': [],
                    'runtimes': [],
                    'runtime_stds': []
                }

            method_results[method]['k'].append(int(k))
            method_results[method]['accuracies'].append(np.mean(mc_results[:, 0]))          # Mean accuracy
            method_results[method]['accuracy_stds'].append(np.std(mc_results[:, 0]))        # Std deviation of accuracy
            method_results[method]['f1_scores'].append(np.mean(mc_results[:, 1]))           # Mean F1 score
            method_results[method]['f1_score_stds'].append(np.std(mc_results[:, 1]))        # Std deviation of F1 score
            method_results[method]['runtimes'].append(np.mean(mc_results[:, 2]))            # Mean runtime
            method_results[method]['runtime_stds'].append(np.std(mc_results[:, 2]))         # Std deviation of runtime

        # Create a figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

        # First subplot: Accuracy
        for method, results in method_results.items():
            ax1.plot(results['k'], results['accuracies'], label=method, marker='o')
            ax1.fill_between(results['k'],
                            np.array(results['accuracies']) - np.array(results['accuracy_stds']),
                            np.array(results['accuracies']) + np.array(results['accuracy_stds']),
                            alpha=0.2)

        ax1.set_xlabel('k Dimension')
        ax1.set_ylabel('Accuracy')
        ax1.axhline(baseline_accuracy, color='tab:blue', linestyle='--', label='Baseline Accuracy')
        ax1.set_title('Accuracy vs. k Dimension')
        ax1.legend(loc='upper left')

        # Second subplot: F1 Score
        for method, results in method_results.items():
            ax2.plot(results['k'], results['f1_scores'], label=method, marker='o')
            ax2.fill_between(results['k'],
                            np.array(results['f1_scores']) - np.array(results['f1_score_stds']),
                            np.array(results['f1_scores']) + np.array(results['f1_score_stds']),
                            alpha=0.2)

        ax2.set_xlabel('k Dimension')
        ax2.set_ylabel('F1 Score')
        ax2.axhline(baseline_f1_score, color='tab:orange', linestyle='--', label='Baseline F1 Score')
        ax2.set_title('F1 Score vs. k Dimension')
        ax2.legend(loc='upper left')

        # Third subplot: Runtime
        for method, results in method_results.items():
            ax3.plot(results['k'], results['runtimes'], label=method, marker='o', linestyle='--')
            ax3.fill_between(results['k'],
                            np.array(results['runtimes']) - np.array(results['runtime_stds']),
                            np.array(results['runtimes']) + np.array(results['runtime_stds']),
                            alpha=0.2)

        ax3.set_xlabel('k Dimension')
        ax3.set_ylabel('Runtime (s)')
        ax3.axhline(baseline_runtime, color='black', linestyle='--', label='Baseline Runtime')
        ax3.set_title('Runtime vs. k Dimension')
        ax3.legend(loc='upper left')

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(outpath)
        print("saved to ",outpath)
        plt.close()


import matplotlib.cm as cm

# def plot_purity_score_of_c_cluster_same_as_original_targets(purity_scores, outpath, tag):
#     savepath = f'{outpath}/{tag}purity_score_graph.png'
#     if not os.path.exists(savepath):
#         # Initialize dictionaries to store data by reduction method and clustering algorithm
#         kmeans_data = {method: [] for method in DIMENSION_REDUCE_METHODS}
#         gmm_data = {method: [] for method in DIMENSION_REDUCE_METHODS}
#         dimensions = sorted(set(int(key.split('_')[1][:-1]) for key in purity_scores.keys()))

        
#         # Populate kmeans and gmm data with purity scores by dimension
#         for key, score in purity_scores.items():
#             method, dim, algo = key.split('_')[0], int(key.split('_')[1][:-1]), key.split('_')[2]
#             if algo == 'kmeans':
#                 kmeans_data[method].append((dim, score))
#             elif algo == 'gmm':
#                 gmm_data[method].append((dim, score))

#         # Sort the data by dimensions
#         for method in DIMENSION_REDUCE_METHODS:
#             kmeans_data[method].sort(key=lambda x: x[0])
#             gmm_data[method].sort(key=lambda x: x[0])

#         # Define colors for each method
#         colors = cm.get_cmap('tab10', len(DIMENSION_REDUCE_METHODS))

#         # Plot with a single y-axis
#         fig, ax1 = plt.subplots(figsize=(10, 6))

#         # Plot kmeans and gmm data on the same axis with consistent colors and different line styles
#         for i, method in enumerate(DIMENSION_REDUCE_METHODS):
#             dims_kmeans, scores_kmeans = zip(*kmeans_data[method])
#             dims_gmm, scores_gmm = zip(*gmm_data[method])
            
#             # Use solid line for KMeans and dashed line for GMM, with the same color per method
#             ax1.plot(dims_kmeans, scores_kmeans, label=f'{method} (KMeans)', color=colors(i), linestyle='-', marker='o')
#             ax1.plot(dims_gmm, scores_gmm, label=f'{method} (GMM)', color=colors(i), linestyle='--', marker='x')

#         ax1.set_xlabel('Number of Dimensions')
#         ax1.set_ylabel('Purity Score')
#         ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

#         plt.title('Purity Score of Clustering Results by Dimension Reduction Method and Clustering Algorithm')
#         fig.tight_layout()
#         plt.savefig(savepath)
#         plt.close()


# def plot_purity_score_of_c_cluster_same_as_original_targets(purity_scores, outpath, tag):
#     savepath = f'{outpath}/{tag}purity_score_graph.png'
#     if not os.path.exists(savepath):
#         # Initialize dictionaries to store data by reduction method and clustering algorithm
#         kmeans_data = {method: [] for method in DIMENSION_REDUCE_METHODS}
#         gmm_data = {method: [] for method in DIMENSION_REDUCE_METHODS}
#         dimensions = sorted(set(int(key.split('_')[1][:-1]) for key in purity_scores.keys()))

        
#         # Populate kmeans and gmm data with purity scores by dimension
#         for key, score in purity_scores.items():
#             method, dim, algo = key.split('_')[0], int(key.split('_')[1][:-1]), key.split('_')[2]
#             if algo == 'kmeans':
#                 kmeans_data[method].append((dim, score))
#             elif algo == 'gmm':
#                 gmm_data[method].append((dim, score))

#         # Sort the data by dimensions
#         for method in DIMENSION_REDUCE_METHODS:
#             kmeans_data[method].sort(key=lambda x: x[0])
#             gmm_data[method].sort(key=lambda x: x[0])

#         # Define colors for each method
#         colors = cm.get_cmap('tab10', len(DIMENSION_REDUCE_METHODS))

#         # Plot with a single y-axis
#         fig, ax1 = plt.subplots(figsize=(10, 6))

#         # Plot kmeans and gmm data on the same axis with consistent colors and different line styles
#         for i, method in enumerate(DIMENSION_REDUCE_METHODS):
#             dims_kmeans, scores_kmeans = zip(*kmeans_data[method])
#             dims_gmm, scores_gmm = zip(*gmm_data[method])
            
#             # Use solid line for KMeans and dashed line for GMM, with the same color per method
#             ax1.plot(dims_kmeans, scores_kmeans, label=f'{method} (KMeans)', color=colors(i), linestyle='-', marker='o')
#             ax1.plot(dims_gmm, scores_gmm, label=f'{method} (GMM)', color=colors(i), linestyle='--', marker='x')

#         ax1.set_xlabel('Number of Dimensions')
#         ax1.set_ylabel('Purity Score')
#         ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

#         plt.title('Purity Score of Clustering Results by Dimension Reduction Method and Clustering Algorithm')
#         fig.tight_layout()
#         plt.savefig(savepath)
#         plt.close()

def plot_purity_score_of_c_cluster(purity_scores, outpath, tag):
    savepath = f'{outpath}/{tag}purity_score_graph.png'
    if not os.path.exists(savepath):

        print(purity_scores)
        # Identify unique clustering algorithms in the purity_scores keys
        cluster_algos = sorted(set(key.split('_')[2] for key in purity_scores.keys()))

        # Initialize a dictionary to store data by reduction method and clustering algorithm
                # Dynamically get dimension reduction methods from purity_scores
        methods = sorted(set(key.split('_')[0] for key in purity_scores.keys()))
        data = {algo: {method: [] for method in methods} for algo in cluster_algos}

        dimensions = sorted(set(int(key.split('_')[1][:-1]) for key in purity_scores.keys()))

        # Populate data for each clustering algorithm and dimension reduction method
        for key, score in purity_scores.items():
            method, dim, algo = key.split('_')[0], int(key.split('_')[1][:-1]), key.split('_')[2]
            data[algo][method].append((dim, score))

        # Sort each method's data by dimension within each clustering algorithm
        for algo in cluster_algos:
            for method in DIMENSION_REDUCE_METHODS:
                data[algo][method].sort(key=lambda x: x[0])

        # Define colors for each dimension reduction method
        colors = cm.get_cmap('tab10', len(DIMENSION_REDUCE_METHODS))

        print("here")
        print(methods)
        print(dimensions)

        # Create a subplot for each clustering algorithm
        fig, axes = plt.subplots(1, len(cluster_algos), figsize=(7 * len(cluster_algos), 6), sharey=True)

        # Ensure axes is iterable in case there’s only one subplot
        if len(cluster_algos) == 1:
            axes = [axes]

        # Plot data for each clustering algorithm in its respective subplot
        for ax, algo in zip(axes, cluster_algos):
            for i, method in enumerate(DIMENSION_REDUCE_METHODS):
                dims, scores = zip(*data[algo][method])
                ax.plot(dims, scores, label=f'{method}', color=colors(i), linestyle='-', marker='o')

            # Set labels, title, and legend for each subplot
            ax.set_xlabel('Number of Dimensions')
            ax.set_title(f'Purity Score for {algo.upper()}')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Set the common y-axis label
        axes[0].set_ylabel('Purity Score')

        # Set a common title and save the figure
        plt.suptitle('Purity Score of Clustering Results by Dimension Reduction Method and Clustering Algorithm')
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the main title
        plt.savefig(savepath)
        plt.close()


import matplotlib.pyplot as plt
import numpy as np



from matplotlib import cm  # For color maps

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_3d_comparison(clustered_reduced_results, color_map="RdYlGn", outpath=None):
    save_dir = outpath + f"/comparison_accuracy_f1.png"
    if not os.path.exists(save_dir):
        # Extract baseline values for comparison
        baseline_results = clustered_reduced_results.get("baseline", {}).get('mc_results', [])
        if baseline_results:
            baseline_accuracy_avg = np.mean([fold_metrics[0] for fold_metrics in baseline_results])
            baseline_accuracy_std = np.std([fold_metrics[0] for fold_metrics in baseline_results])
            baseline_f1_avg = np.mean([fold_metrics[1] for fold_metrics in baseline_results])
            baseline_f1_std = np.std([fold_metrics[1] for fold_metrics in baseline_results])

        # Prepare arrays to store the metric values
        dreduc_algos = sorted({key[0] for key in clustered_reduced_results if key != "baseline"})
        cluster_algos = sorted({key[2] for key in clustered_reduced_results if key != "baseline"})
        k_dims = sorted({key[1] for key in clustered_reduced_results if key != "baseline"})
        n_clusters_vals = sorted({key[3] for key in clustered_reduced_results if key != "baseline"})

        # Prepare a grid to store metric averages and masks for each (dreduc_algo, cluster_algo) combination
        accuracy_grid = np.zeros((len(dreduc_algos) + 1, len(cluster_algos) + 1))  # Baseline at (0,0)
        f1_grid = np.zeros_like(accuracy_grid)
        accuracy_texts = np.empty_like(accuracy_grid, dtype="object")
        f1_texts = np.empty_like(f1_grid, dtype="object")
        mask = np.zeros_like(accuracy_grid, dtype=bool)  # Mask to skip zero cells

        # Populate the grids with average ± std values and set masks
        for key, result in clustered_reduced_results.items():
            if key == "baseline":
                continue
            dreduc_algo, k_dim, cluster_algo, n_clusters = key
            accuracy_values = [fold_metrics[0] for fold_metrics in result['mc_results']]
            f1_values = [fold_metrics[1] for fold_metrics in result['mc_results']]
            avg_accuracy = np.mean(accuracy_values)
            std_accuracy = np.std(accuracy_values)
            avg_f1 = np.mean(f1_values)
            std_f1 = np.std(f1_values)

            # Determine grid position
            x = dreduc_algos.index(dreduc_algo) + 1  # +1 to account for baseline at (0,0)
            y = cluster_algos.index(cluster_algo) + 1

            # Assign values to the grid and text arrays
            accuracy_grid[x, y] = avg_accuracy
            f1_grid[x, y] = avg_f1
            accuracy_texts[x, y] = f"{avg_accuracy:.3f} ± {std_accuracy:.3f}"
            f1_texts[x, y] = f"{avg_f1:.3f} ± {std_f1:.3f}"
            mask[x, y] = False  # Ensure cell is visible

        # Set baseline values
        if baseline_results:
            accuracy_grid[0, 0] = baseline_accuracy_avg
            f1_grid[0, 0] = baseline_f1_avg
            accuracy_texts[0, 0] = f"{baseline_accuracy_avg:.3f} ± {baseline_accuracy_std:.3f}"
            f1_texts[0, 0] = f"{baseline_f1_avg:.3f} ± {baseline_f1_std:.3f}"
            mask[0, 0] = False  # Ensure baseline cell is visible

        # Apply mask to all zero cells
        mask[accuracy_grid == 0] = True
        mask[f1_grid == 0] = True

        # Plotting the heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        cmap = sns.color_palette(color_map, as_cmap=True)

        # Accuracy Heatmap
        sns.heatmap(accuracy_grid, annot=accuracy_texts, fmt="", cmap=cmap, mask=mask, ax=ax1, cbar=True)
        ax1.set_title('Accuracy Comparison')
        ax1.set_xlabel('Clustering Algorithms')
        ax1.set_ylabel('Dimensionality Reduction Algorithms')
        ax1.set_xticks(np.arange(len(cluster_algos) + 1))
        ax1.set_xticklabels(["baseline"] + [f"{alg} (n={n})" for alg, n in zip(cluster_algos, n_clusters_vals)], rotation=45)
        ax1.set_yticks(np.arange(len(dreduc_algos) + 1))
        ax1.set_yticklabels(["baseline"] + [f"{alg} (k={k})" for alg, k in zip(dreduc_algos, k_dims)], rotation=45)

        # F1 Score Heatmap
        sns.heatmap(f1_grid, annot=f1_texts, fmt="", cmap=cmap, mask=mask, ax=ax2, cbar=True)
        ax2.set_title('F1 Score Comparison')
        ax2.set_xlabel('Clustering Algorithms')
        ax2.set_ylabel('Dimensionality Reduction Algorithms')
        ax2.set_xticks(np.arange(len(cluster_algos) + 1))
        ax2.set_xticklabels(["baseline"] + [f"{alg} (n={n})" for alg, n in zip(cluster_algos, n_clusters_vals)], rotation=45)
        ax2.set_yticks(np.arange(len(dreduc_algos) + 1))
        ax2.set_yticklabels(["baseline"] + [f"{alg} (k={k})" for alg, k in zip(dreduc_algos, k_dims)], rotation=45)

        plt.tight_layout()

        # Save figure if output path is provided
        if outpath:
            plt.savefig(save_dir)
        print("Saved graph to ", save_dir)
        
        plt.close()



def plot_multi_histograms(clustered_reduced_results, outpath=None):
    save_dir_acc = outpath + f"/accuracy_histograms.png"
    save_dir_f1 = outpath + f"/f1_histograms.png"
    
    if not os.path.exists(save_dir_acc) or not os.path.exists(save_dir_f1):
    
        # Extract baseline results for comparison
        baseline_results = clustered_reduced_results.get("baseline", {}).get('mc_results', [])
        baseline_accuracy = [fold_metrics[0] for fold_metrics in baseline_results]
        baseline_f1 = [fold_metrics[1] for fold_metrics in baseline_results]

        # Unique reduction and clustering algorithms
        dreduc_algos = sorted({key[0] for key in clustered_reduced_results if key != "baseline"})
        cluster_algos = sorted({key[2] for key in clustered_reduced_results if key != "baseline"})
        
        # Prepare figures for accuracy and F1 score histograms
        fig_accuracy, axs_accuracy = plt.subplots(len(dreduc_algos), len(cluster_algos), figsize=(16, 12), sharex=True, sharey=True)
        fig_f1, axs_f1 = plt.subplots(len(dreduc_algos), len(cluster_algos), figsize=(16, 12), sharex=True, sharey=True)

        # Set up baseline histogram colors and labels
        baseline_color = "lightgrey"
        baseline_label = "Baseline"

        # Iterate over dimensionality reduction and clustering algorithm combinations
        for i, dreduc_algo in enumerate(dreduc_algos):
            for j, cluster_algo in enumerate(cluster_algos):
                # Collect accuracy and F1 values for all (k_dim, n_clusters) under each (dreduc_algo, cluster_algo)
                accuracy_values = []
                f1_values = []

                for key, result in clustered_reduced_results.items():
                    if key != "baseline" and isinstance(key, tuple) and len(key) == 4:
                        dim_red, k_dim, clust_alg, n_clusters = key
                        if dim_red == dreduc_algo and clust_alg == cluster_algo:
                            acc_values = [fold_metrics[0] for fold_metrics in result['mc_results']]
                            f1_vals = [fold_metrics[1] for fold_metrics in result['mc_results']]
                            accuracy_values.append((k_dim, n_clusters, acc_values))
                            f1_values.append((k_dim, n_clusters, f1_vals))

                # Accuracy Histogram
                ax_acc = axs_accuracy[i, j]
                for k_dim, n_clusters, acc_vals in accuracy_values:
                    ax_acc.hist(acc_vals, bins=10, alpha=0.6, label=f"k={k_dim}, n={n_clusters}")
                ax_acc.hist(baseline_accuracy, bins=10, alpha=0.3, color=baseline_color, label=baseline_label)
                ax_acc.set_title(f"{dreduc_algo} + {cluster_algo} (Accuracy)")
                ax_acc.set_xlabel("Accuracy")
                ax_acc.set_ylabel("Frequency")
                # ax_acc.legend()

                # F1 Histogram
                ax_f1 = axs_f1[i, j]
                for k_dim, n_clusters, f1_vals in f1_values:
                    ax_f1.hist(f1_vals, bins=10, alpha=0.6, label=f"k={k_dim}, n={n_clusters}")
                ax_f1.hist(baseline_f1, bins=10, alpha=0.3, color=baseline_color, label=baseline_label)
                ax_f1.set_title(f"{dreduc_algo} + {cluster_algo} (F1)")
                ax_f1.set_xlabel("F1 Score")
                ax_f1.set_ylabel("Frequency")
                # ax_f1.legend()

        # Adjust layout for clarity
        fig_accuracy.tight_layout()
        fig_f1.tight_layout()

        # Save figures if output path is provided
        if outpath:
            fig_accuracy.savefig(save_dir_acc)
            fig_f1.savefig(save_dir_f1)
        
        plt.close()



import glob

def plot_purity_significance_from_pkl(outpath=None):
    # Define the output path for the saved figure
    save_dir = outpath + f"/purity_score_statis_significance.png"
    if not os.path.exists(save_dir):
    
        # Find all .pkl files with p-value results
        pkl_files = glob.glob(f"{DREDUCED_CLUSTER_PKL_OUTDIR}/p_value_results_*.pkl")
        
        # Dictionary to store results for plotting
        plot_data = {}

        for pkl_file in pkl_files:
            # Extract threshold from file name
            threshold = float(pkl_file.split('_')[-1][:-4])

            # Load the data
            with open(pkl_file, 'rb') as file:
                p_value_results = pickle.load(file)
            
            # Prepare data for each algorithm and method
            for algo, methods in p_value_results.items():
                for method, result in methods.items():
                    avg_improvement = result['avg_improvement']
                    p_value = result['p_value']
                    significant = result['significant']

                    if algo not in plot_data:
                        plot_data[algo] = {}
                    if method not in plot_data[algo]:
                        plot_data[algo][method] = {'thresholds': [], 'p_values': [], 'significance': []}
                    
                    plot_data[algo][method]['thresholds'].append(threshold)
                    plot_data[algo][method]['p_values'].append(p_value)
                    plot_data[algo][method]['significance'].append(significant)
        method_colors = {'PCA': 'blue', 'ICA': 'orange', 'RCA': 'green'}  # Example mapping, customize as needed


        ##########################################      OG           
        # Create a single figure with multiple subplots in a single row
        n_algos = len(plot_data)
        ncols = n_algos  # Number of columns equals the number of algorithms
        fig, axes = plt.subplots(1, ncols, figsize=(6 * n_algos, 6))  # Create subplots in a single row

        # If there is only one subplot, `axes` is a single axis instead of an array
        if n_algos == 1:
            axes = [axes]

        # Define a color palette for the methods
        method_colors = {'PCA': 'blue', 'ICA': 'orange', 'RCA': 'green'}  # Example mapping, customize as needed

        for i, (algo, methods) in enumerate(plot_data.items()):
            ax = axes[i]  # Get the current axis for the algorithm's subplot
            for method, results in methods.items():
                thresholds = results['thresholds']
                p_values = results['p_values']
                significance = results['significance']
                
                # Get the color for the current method
                color = method_colors.get(method, 'black')  # Default to black if method not in the color map
                
                # Plot p-values with color based on significance
                colors = [color for sig in significance]  # Use method color for all points
                ax.scatter(thresholds, p_values, label=method, color=colors)
            
            # Add significance threshold line (p=0.05), color it green
            ax.axhline(y=0.05, color='green', linestyle='-', label='Significance Threshold (p=0.05)')
            
            # Set labels and title for each subplot
            ax.set_xlabel('Improvement Threshold')
            ax.set_ylabel('P-Value')
            ax.set_title(f'Statistical Significance of Purity Score Improvements ({algo})')
            ax.legend(loc='upper left', )


        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Save the figure with all subplots to a single file
        plt.savefig(save_dir)
        plt.close()



def graph_class_imbalance_multilabel(Y_df, outpath):
    start_time = time.time()

    # Calculate the class distribution (percentage) for each label (column) in Y_df
    class_0_counts = (Y_df == 0).sum(axis=0)  # Count of 0s in each column
    class_1_counts = (Y_df == 1).sum(axis=0)  # Count of 1s in each column

    # Calculate the percentage of each class for each label
    class_0_percentages = class_0_counts / Y_df.shape[0]
    class_1_percentages = class_1_counts / Y_df.shape[0]

    # Create a stacked bar chart
    plt.figure(figsize=(12, 6))

    # Stacked bars: class_0_percentages for the bottom part, and class_1_percentages on top
    bar_width = 0.35  # Bar width
    labels = Y_df.columns  # Labels (for the x-ticks)
    
    plt.bar(labels, class_0_percentages, label='Class 0', width=bar_width, color='lightblue')
    plt.bar(labels, class_1_percentages, bottom=class_0_percentages, label='Class 1', width=bar_width, color='orange')

    # Add labels, title, and legend
    plt.xlabel('Labels', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.title('Class Distribution (0 vs 1) for Each Label (Stacked)', fontsize=14)
    plt.legend(title='Class', loc='upper right')

    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90)  # Rotate for better readability
    # Ensure that the y-axis is scaled from 0 to 1
    plt.ylim(0, 1)

    # Add the time to the graph
    end_time = time.time()

    # Tight layout for better presentation
    plt.tight_layout()

    # Save the plot
    plt.savefig(outpath)
    plt.close()

    print(f"Time to graph class imbalance (multi-label): {end_time - start_time:.4f}s")

def graph_feature_heatmap_multilabel(X_df, Y_df, outpath):
    start_time = time.time()
    
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(X_df, rowvar=False)
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", square=True, cbar=True)
    plt.title("Correlation Heatmap of Features in X")
    
    # Save the heatmap
    plt.tight_layout()  # Adjust the layout to prevent clipping
    plt.savefig(outpath)
    plt.close()  # Close the figure

    end_time = time.time()
    print(f"Time to graph feature heatmap (multi-label): {end_time - start_time:.4f}s")
