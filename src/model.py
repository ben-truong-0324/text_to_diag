import utils
import etl, cv
import numpy as np
import pandas as pd
import re
import time
import matplotlib
import os
from datetime import datetime
import unittest


matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import *

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, learning_curve, validation_curve, RandomizedSearchCV, StratifiedKFold

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, log_loss,make_scorer, matthews_corrcoef
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
# from yellowbrick.classifier import ConfusionMatrix
# from yellowbrick.classifier import ROCAUC


# def plot_validation_curve(model, X_train, Y_train, param_name, param_range, outpath, metric = "accuracy"):
# 	"""Plot and save Validation Curve"""
# 	train_scores, test_scores = validation_curve(model, X_train, Y_train, param_name=param_name, param_range=param_range, cv=5, n_jobs=-1, scoring=metric)

# 	# Calculate mean and standard deviation
# 	train_scores_mean = np.mean(train_scores, axis=1)
# 	test_scores_mean = np.mean(test_scores, axis=1)
		
# 	print(train_scores_mean)
# 	print(test_scores_mean)
# 	print(param_range)

# 	# Plot validation curve
# 	plt.figure()
# 	plt.plot(param_range, train_scores_mean, label='Training score', marker='o', markersize=8, linestyle='-')
# 	plt.plot(param_range, test_scores_mean, label='Cross-validation score', marker='o', markersize=8, linestyle='-')
# 	plt.title(f'Validation Curve for {param_name} {metric}')
# 	plt.xlabel(param_name)
# 	plt.ylabel('Score')
# 	plt.legend(loc='best')

# 	# Save validation curve to file
# 	plt.savefig(outpath)
# 	plt.close()

def evaluate_models_all_metrcis_test_only(models, X_train, Y_train, X_test, Y_test, outpath, model_names):
	# Initialize lists to store metrics
	accuracies = []
	aucs = []
	precisions = []
	recalls = []
	f1scores = []
	mccs = []

	# Evaluate each model
	for model in models:
		model.fit(X_train, Y_train)
		Y_pred = model.predict(X_test)
		# Calculate metrics
		acc = accuracy_score(Y_test, Y_pred)
		auc = roc_auc_score(Y_test, Y_pred)
		precision = precision_score(Y_test, Y_pred)
		recall = recall_score(Y_test, Y_pred)
		f1score = f1_score(Y_test, Y_pred)
		mcc = matthews_corrcoef(Y_test, Y_pred)

		# Append metrics to lists
		accuracies.append(acc)
		aucs.append(auc)
		precisions.append(precision)
		recalls.append(recall)
		f1scores.append(f1score)
		mccs.append(mcc)

	# Set up the bar plot
	bar_width = 0.1
	x = np.arange(len(model_names))

	# Create bars for each metric
	plt.figure(figsize=(12, 6))
	plt.bar(x - 2 * bar_width, accuracies, width=bar_width, label='Accuracy')
	plt.bar(x - bar_width, aucs, width=bar_width, label='AUC')
	plt.bar(x, precisions, width=bar_width, label='Precision')
	plt.bar(x + bar_width, recalls, width=bar_width, label='Recall')
	plt.bar(x + 2 * bar_width, f1scores, width=bar_width, label='F1 Score')
	plt.bar(x + 3 * bar_width, mccs, width=bar_width, label='MCC')

	# Adding labels and title
	plt.xlabel('Models')
	plt.ylabel('Metric Values')
	plt.title(f'Performance Metrics for Each Model | {DATASET_SELECTION}')
	plt.xticks(x, model_names)
	plt.legend()

	plt.tight_layout()
	plt.savefig(outpath)  # Save the plot
	plt.show()


def plot_confusion_matrix(model, X_test, Y_test, outpath, model_name):
	"""Plot and save Confusion Matrix"""
	# Predict values
	Y_pred = model.predict(X_test)

	# Compute confusion matrix
	cm = confusion_matrix(Y_test, Y_pred)

	# Plot confusion matrix
	fig, ax = plt.subplots()
	cax = ax.matshow(cm, cmap=plt.cm.Blues)
	plt.title('Confusion Matrix')
	plt.colorbar(cax)
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
	for (i, j), value in np.ndenumerate(cm):
		ax.text(j, i, f'{value}', ha='center', va='center', color='white' if value > cm.max() / 2 else 'black')

	# Save confusion matrix to file
	plt.savefig(outpath)
	print(f"created graph at {outpath}")
	plt.close()

def plot_confusion_matrix_all_models(models, X_train, Y_train, X_test, Y_test, outpath, model_names):
	"""Plot confusion matrices for all models as subplots"""

	# Number of models
	num_models = len(models)

	# Create subplots: Arrange models in grid (2 columns for example, adjust as necessary)
	fig, axs = plt.subplots(3, 2, figsize=(12 , 10))  # Adjust size based on number of metrics
	axs = axs.flatten()

	# Iterate through models
	for i, model in enumerate(models):
		# Fit model to training data
		model.fit(X_train, Y_train)
		
		# Predict on test data
		Y_pred = model.predict(X_test)
		
		# Compute confusion matrix
		cm = confusion_matrix(Y_test, Y_pred)
		
		# Plot confusion matrix
		ax = axs[i]  # Select subplot
		cax = ax.matshow(cm, cmap=plt.cm.Blues)
		ax.set_title(f'{model_names[i]} - Confusion Matrix')
		fig.colorbar(cax, ax=ax)  # Add colorbar
		
		ax.set_xlabel('Predicted Label')
		ax.set_ylabel('True Label')
		
		# Annotate the confusion matrix with counts
		for (row, col), value in np.ndenumerate(cm):
			ax.text(col, row, f'{value}', ha='center', va='center', color='white' if value > cm.max() / 2 else 'black')

	# Hide any unused subplots (in case of an odd number of models)
	for j in range(i + 1, len(axs)):
		fig.delaxes(axs[j])

	# Adjust layout and save the figure
	plt.tight_layout()
	plt.savefig(outpath)
	print(f"created graph at {outpath}")
	plt.close()

# def plot_confusion_matrix_output(model, X_test, Y_test, ax, model_name):
#     disp = plot_confusion_matrix(model, X_test, Y_test, ax=ax, cmap=plt.cm.Blues, values_format='d')
#     ax.set_title(f'Confusion Matrix - {model_name}')


# Function to plot Precision-Recall Curve
def plot_precision_recall_curve_output(model, X_test, Y_test, ax, model_name):
    if hasattr(model, "predict_proba"):
        Y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(Y_test, Y_proba)
        avg_precision = average_precision_score(Y_test, Y_proba)
        ax.plot(recall, precision, color='blue', lw=2, label='AP = %0.2f' % avg_precision)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.legend(loc="lower left")

# 4. Box Plot (for errors)
def plot_box_plot_residuals(model, X_test, Y_test, ax, model_name):
    Y_pred = model.predict(X_test)
    residuals = Y_test - Y_pred
    sns.boxplot(data=residuals, ax=ax)
    ax.set_title(f'Residual Box Plot - {model_name}')
    ax.set_xlabel('Residuals')

# Main function to iterate through models and plot subplots
def plot_model_metrics_output(models, model_names, X_train, X_test, Y_train, Y_test):
	for i, model in enumerate(models):
		# Fit the model
		model.fit(X_train, Y_train)
		# Plot Confusion Matrix
		cm_outpath = f'{OUTPUT_DIR}/CM_{model}_{DATASET_SELECTION}.png'
		plot_confusion_matrix(model, X_test, Y_test, cm_outpath, model_names[i])

def plot_validation_curve_by_param_and_metric(model, model_name, X_train, Y_train, param_name, param_range, outpath, metrics):
	num_metrics = len(metrics)

	# Set up a figure with one subplot for each metric
	fig, axs = plt.subplots(3, 2, figsize=(12 , 10))  # Adjust size based on number of metrics
	axs = axs.flatten()

	fig.suptitle(f"Validation Curves - {model_name}\n{param_name}", fontsize=16)

	if num_metrics == 1:  # In case only one metric, axs is not an array
		axs = [axs]

	x_range = range(len(param_range))

	for i, metric in enumerate(metrics):
		# Compute the validation curve for the given metric
		train_scores, test_scores = validation_curve(model, X_train, Y_train, param_name=param_name, param_range=param_range, cv=5, n_jobs=-1, scoring=metric)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		
		ax = axs[i]  # Select the subplot axis for the current metric
		
		ax.plot(x_range, train_scores_mean, label="Training score", color="darkorange", lw=2, marker='o', markersize=8)
		ax.plot(x_range, test_scores_mean, label="Cross-validation score", color="navy", lw=2, marker='o', markersize=8)

		# Fill between the curves to represent standard deviation
		ax.fill_between(x_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=2)
		ax.fill_between(x_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=2)
		
		# Set the title and labels for the subplot
		ax.set_title(f"Metric: {metric}")
		ax.set_xlabel(param_name)
		ax.set_ylabel(metric)
		ax.set_ylim(0.0, 1.1)
		ax.set_xticks(x_range)
		ax.set_xticklabels([str(val) for val in param_range], rotation=45, ha='right')
		ax.legend(loc="best")

	# Adjust layout to prevent overlap
	plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle

	# Save the entire figure with all subplots
	plt.savefig(outpath)
	print(f"created graph at {outpath}")
	plt.close()

def plot_validation_curve(model, model_name, X_train, Y_train, param_name, param_range, outpath, metric = "accuracy"):


	train_scores, test_scores = validation_curve(model, X_train, Y_train, param_name=param_name, param_range=param_range, cv=5, n_jobs=-1, scoring=metric)

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	x_range = range(len(param_range))

	plt.figure(figsize=(12, 6))
	plt.title(f"Validation Curve - {model_name}\n{param_name}")
	plt.xlabel(param_name)
	plt.ylabel(metric)
	plt.ylim(0.0, 1.1)

	plt.plot(x_range, train_scores_mean, label="Training score",
				color="darkorange", lw=2, marker='o', markersize=8)
	plt.plot(x_range, test_scores_mean, label="Cross-validation score",
				color="navy", lw=2, marker='o', markersize=8)

	plt.fill_between(x_range, train_scores_mean - train_scores_std,
						train_scores_mean + train_scores_std, alpha=0.2,
						color="darkorange", lw=2)
	plt.fill_between(x_range, test_scores_mean - test_scores_std,
						test_scores_mean + test_scores_std, alpha=0.2,
						color="navy", lw=2)

	plt.xticks(x_range, [str(val) for val in param_range], rotation=45, ha='right')
	plt.tight_layout()
	plt.legend(loc="best")

	plt.savefig(outpath)
	print(f"created graph at {outpath}")
	plt.close()

def graph_validation_curves(models, model_names,X_df, Y_df, metric):
	# Define parameter distributions for each model
	param_distributions = PARAM_DISTRIBUTIONS_SEARCHCV 
	# Initialize an empty list to store optimized models
	optimized_models = []
	for model, name in zip(models, model_names):
		for param_name, param_range in param_distributions[name].items():
			print(f"Plotting validation curves for {name} {metric} {param_name}...")			
			start_time = time.time()
			outpath = f'{OUTPUT_DIR}/validation_curve_{model}_{param_name}_{metric}_{DATASET_SELECTION}.png'
			plot_validation_curve(model, name, X_df, Y_df, param_name, param_range, outpath, metric)

			outpath = f'{OUTPUT_DIR}/validation_curve_all_metrics_{model}_{param_name}_{DATASET_SELECTION}.png'
			metrics = [
						"accuracy","precision","recall","f1","roc_auc","average_precision",   #for classification
						# "neg_mean_absolute_error","neg_mean_squared_error","r2","log_loss", #for regression
						]
			plot_validation_curve_by_param_and_metric(model, name, X_df, Y_df, param_name, param_range, outpath, metrics)
			end_time = time.time()
			print(("Time to plot validation curve: " + str(end_time - start_time) + "s"))

def runRandomizedSearch(models, model_names,X_df, Y_df, metric):
	# Define parameter distributions for each model
	param_distributions = PARAM_DISTRIBUTIONS_SEARCHCV 
	# Initialize an empty list to store optimized models
	optimized_models = []
	for model, name in zip(models, model_names):
		print(f"Optimizing hyperparameters for {name} {metric}...")
		if name == 'NN':
			search = RandomizedSearchCV(model, param_distributions['NN'], n_iter=10, random_state=GT_ID, cv=5, scoring=metric)
		elif name == 'SVM Linear':
			search = RandomizedSearchCV(model, param_distributions['SVM Linear'], n_iter=10, random_state=GT_ID, cv=5, scoring=metric)
		elif name == 'SVM RBF':
			search = RandomizedSearchCV(model, param_distributions['SVM RBF'], n_iter=10, random_state=GT_ID, cv=5, scoring=metric)
		elif name == 'KNN':
			search = RandomizedSearchCV(model, param_distributions['KNN'], n_iter=10, random_state=GT_ID, cv=5, scoring=metric)
		elif name == 'DT Boosted':
			search = RandomizedSearchCV(model, param_distributions['DT Boosted'], n_iter=10, random_state=GT_ID, cv=5, scoring=metric)
		elif name == 'RF':
			search = RandomizedSearchCV(model, param_distributions['RF'], n_iter=10, random_state=GT_ID, cv=5, scoring=metric)
		# Fit the model and find the best parameters
		search.fit(X_df, Y_df)
		optimized_models.append(search.best_estimator_)
		#note it down in new file
		with open(f"./rndSx_{datetime.today().strftime('%Y-%m-%d')}.txt", 'a') as f:
			f.write(f"Best parameters for {name} ({metric}): {search.best_params_}\n")


def specificity(y_true, y_pred):
    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Check if confusion matrix has the right shape
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else np.nan  # Avoid division by zero
    else:
        return np.nan  


def plot_combined_learning_curves(models, model_names, X_train, Y_train, metric, is_strat, outpath):
	"""Plot and save learning curves for multiple models."""
	n_models = len(models)
	fig, axes = plt.subplots(3, 2, figsize=(12, 10))  # Adjust the size as needed
	axes = axes.flatten()  # Flatten to easily iterate

	for i, model in enumerate(models):
		if is_strat:
			cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=GT_ID)
		else:
			cv = 5

		if metric == "specificity":
			specificity_scorer = make_scorer(specificity)
			train_sizes, train_scores, test_scores = learning_curve(
				model, X_train, Y_train, cv=cv, n_jobs=-1, scoring=specificity_scorer
			)
		else:
			train_sizes, train_scores, test_scores = learning_curve(
				model, X_train, Y_train, cv=cv, n_jobs=-1, scoring=metric,
			)

		# Calculate mean and standard deviation
		train_scores_mean = np.mean(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)


		# Plot learning curves
		axes[i].plot(train_sizes, train_scores_mean, label='Training score', color='blue', marker='o', markersize=8, linestyle='-')
		axes[i].plot(train_sizes, test_scores_mean, label='Cross-validation score', color='green', marker='o', markersize=8, linestyle='-')

		# Fill between the lines for standard deviation
		axes[i].fill_between(train_sizes, 
								train_scores_mean - train_scores_std, 
								train_scores_mean + train_scores_std, 
								color='blue', alpha=0.2)
		axes[i].fill_between(train_sizes, 
								test_scores_mean - test_scores_std, 
								test_scores_mean + test_scores_std, 
								color='green', alpha=0.2)

		# Set titles and labels
		axes[i].set_title(f'Dataset: {DATASET_SELECTION} | Metric: {metric}\nLearning Curve ({model_names[i]})')
		axes[i].set_xlabel('Training Examples')
		axes[i].set_ylabel('Score')
		axes[i].legend(loc='best')

	plt.tight_layout()  # Adjust layout
	plt.savefig(outpath,format='png', dpi=300) # Save the figure
	print(f"created graph at {outpath}")
	plt.close()





def main():
	print("hello world, my_model.py here")

	# Create the directory if it doesn't exist
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	X_df, Y_df = etl.get_data(DATASET_SELECTION, 1, 0)

	# Split the dataset into training and testing sets
	X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=GT_ID)

	if 'credit' in DATASET_SELECTION:
		models = MODELS_CREDIT_DATASET
	elif 'gps' in DATASET_SELECTION:
		models = MODELS_GPS_DATASET
	
	model_names = MODEL_NAMES
	metric = ['accuracy', 'f1', 'precision', 'roc_auc', 'specificity']
	#################### get combined LC graphs
	for m in metric:
		print("getting learning curves for ",m)
		start_time = time.time()
		plot_combined_learning_curves(models, model_names, X_df, Y_df, metric = m, is_strat = 0, outpath = f'{OUTPUT_DIR}/COMBINED_{m}_{DATASET_SELECTION}_learning_curves.png')
		plot_combined_learning_curves(models, model_names, X_df, Y_df, metric = m, is_strat = 1, outpath = f'{OUTPUT_DIR}/COMBINED_stratCV_{m}_{DATASET_SELECTION}_learning_curves.png')
		end_time = time.time()
		print(("Time to plot combined learning curves: " + str(end_time - start_time) + "s"))
	
	####################m loss curve for NN
	# print("getting loss curve ")
	# start_time = time.time()
	# end_time = time.time()
	# print(("Time to plot loss curves: " + str(end_time - start_time) + "s"))

	#################### hyperparam optimization and validation curves
	start_time = time.time()
	for m in SEARCHCV_METRICS_CLASSIFICATION:
		runRandomizedSearch(models, model_names,X_df, Y_df, m)
		graph_validation_curves(models, model_names,X_df, Y_df, m)
	end_time = time.time()
	print(("Time to runRandomizedSearch: " + str(end_time - start_time) + "s"))

	#################### models cm output
	plot_model_metrics_output(models, model_names, X_train, X_test, Y_train, Y_test)
	cm_all_outpath = f'{OUTPUT_DIR}/ALL_CM_{DATASET_SELECTION}.png'
	plot_confusion_matrix_all_models(models, X_train, Y_train, X_test, Y_test, cm_all_outpath, model_names)

	#################### models metrics output
	print("getting all metrics, test set only")
	start_time = time.time()
	metric_all_outpath_test_only = f'{OUTPUT_DIR}/ALL_METRIC_Test_{DATASET_SELECTION}.png'
	evaluate_models_all_metrcis_test_only(models, X_train, Y_train, X_test, Y_test, metric_all_outpath_test_only, model_names)
	end_time = time.time()
	print(("Time to plot test set all metrcis: " + str(end_time - start_time) + "s"))

	print("getting all metrics, kfCV for training set")
	start_time = time.time()
	metric_all_outpath_kfold = f'{OUTPUT_DIR}/ALL_METRIC_kFCV_{DATASET_SELECTION}.png'
	cv.evaluate_models_all_metrcis(models, X_df, Y_df, metric_all_outpath_kfold, model_names, 1, 0, 5) #with kfold
	end_time = time.time()
	print(("Time to plot kFCV all metrcis: " + str(end_time - start_time) + "s"))

	print("getting all metrics, MCCV for training set")
	start_time = time.time()
	metric_all_outpath_mccv = f'{OUTPUT_DIR}/ALL_METRIC_MCCV_{DATASET_SELECTION}.png'
	cv.evaluate_models_all_metrcis(models, X_df, Y_df, metric_all_outpath_mccv, model_names, 0, 1, 5) #with mccv
	end_time = time.time()
	print(("Time to plot MCCV all metrcis: " + str(end_time - start_time) + "s"))


if __name__ == "__main__":
	main()
