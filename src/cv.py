import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import etl  # assuming etl.py is properly imported
from yellowbrick.model_selection import FeatureImportances
from config import *

# Function to compute classification metrics
# def classification_metrics(Y_true, Y_pred):
# 	acc = accuracy_score(Y_true, Y_pred)
# 	auc_ = roc_auc_score(Y_true, Y_pred) if len(set(Y_true)) > 1 else 'N/A'
# 	precision = precision_score(Y_true, Y_pred, average='weighted')
# 	recall = recall_score(Y_true, Y_pred, average='weighted')
# 	f1score = f1_score(Y_true, Y_pred, average='weighted')
# 	return acc, auc_, precision, recall, f1score

def classification_metrics(Y_true, Y_pred, Y_prob):
    acc = accuracy_score(Y_true, Y_pred)
    auc_ = roc_auc_score(Y_true, Y_prob)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    f1score = f1_score(Y_true, Y_pred)
    mcc = matthews_corrcoef(Y_true, Y_pred)  # Calculate MCC

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero

    return acc, auc_, precision, recall, f1score, mcc, specificity

def kfold_cv_all_metrics(model, kf, X, Y, accuracies, aucs,precisions,recalls,f1scores,mccs,specificities):
	for train_index, test_index in kf.split(X):
		try:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]
		except:
			X_train, X_test = X.iloc[train_index], X.iloc[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

		# Fit the model and predict
		Y_pred = model.fit(X_train, Y_train).predict(X_test)
		Y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for AUC
		
		# Calculate metrics
		acc, auc_, precision, recall, f1score, mcc, specificity= classification_metrics(Y_test, Y_pred, Y_prob)

		accuracies.append(acc)
		aucs.append(auc_)
		precisions.append(precision)
		recalls.append(recall)
		f1scores.append(f1score)
		mccs.append(mcc)
		specificities.append(specificity)
	return accuracies, aucs,precisions,recalls,f1scores,mccs,specificities

def mccv_all_metrics(model, kf, X, Y, accuracies, aucs,precisions,recalls,f1scores,mccs,specificities):
	np.random.seed(GT_ID)
	for _ in range(MONTE_CARLO_CV_ITER):
		# Randomly split the data
		indices = np.random.permutation(len(X))
		train_size = int(len(X) * MONTE_CARLO_CV_TRAIN_SIZE)
		train_indices = indices[:train_size]
		test_indices = indices[train_size:]

		try:
			X_train, X_test = X[train_indices], X[test_indices]
			Y_train, Y_test = Y[train_indices], Y[test_indices]
		except:
			X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
			Y_train, Y_test = Y[train_indices], Y[test_indices]

		# Fit the model and predict
		Y_pred = model.fit(X_train, Y_train).predict(X_test)
		Y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for AUC
		
		# Calculate metrics
		acc, auc_, precision, recall, f1score, mcc, specificity= classification_metrics(Y_test, Y_pred, Y_prob)

		# Append original metric values
		accuracies.append(acc)
		aucs.append(auc_)
		precisions.append(precision)
		recalls.append(recall)
		f1scores.append(f1score)
		mccs.append(mcc)
		specificities.append(specificity)
	return accuracies, aucs,precisions,recalls,f1scores,mccs, specificities


def evaluate_models_all_metrcis(models, X, Y, outpath, model_names, with_kfold, with_mccv, k=5):
	kf = KFold(n_splits=k, shuffle=True, random_state=GT_ID)

	# Initialize lists to hold metrics for each model
	metrics = {
		'Model': [],
		'Accuracy': [],
		'AUC': [],
		'Precision': [],
		'Specificity': [],  # Updated to include Specificity
		'Recall': [],
		'F1 Score': [],
		'MCC': []
	}

	for model in models:
		accuracies = []
		aucs = []
		precisions = []
		specificities = [] 
		recalls = []
		f1scores = []
		mccs = []
		#####
		if with_kfold:
			accuracies, aucs,precisions,recalls,f1scores,mccs, specificities = kfold_cv_all_metrics(model, kf, X, Y, accuracies, aucs,precisions,recalls,f1scores,mccs,specificities)
		elif with_mccv:
			accuracies, aucs,precisions,recalls,f1scores,mccs, specificities = mccv_all_metrics(model, kf, X, Y, accuracies, aucs,precisions,recalls,f1scores,mccs,specificities)

		#######

		# Store average metrics for each model
		metrics['Model'].append(model_names[len(metrics['Model'])])
		metrics['Accuracy'].append(accuracies)
		metrics['AUC'].append(aucs)
		metrics['Precision'].append(precisions)
		metrics['Recall'].append(recalls)
		metrics['F1 Score'].append(f1scores)
		metrics['Specificity'].append(specificities)
		metrics['MCC'].append(mccs)

	# Convert metrics to DataFrame for easier plotting
	metrics_df = pd.DataFrame(metrics)

	# Create subplots for each metric
	metrics_list = ['Accuracy', 'AUC', 
						'Precision', 
						'Recall',
						# 'Specificity',
						'F1 Score', 'MCC']
	n_metrics = len(metrics_list)

	fig, axes = plt.subplots(3, 2, figsize=(12, 10))
	axes = axes.flatten()

	for i, metric in enumerate(metrics_list):
		sns.boxplot(data=metrics_df.explode(metric), x='Model', y=metric, ax=axes[i])
		axes[i].set_title(metric)
		axes[i].set_xlabel('Models')
		axes[i].set_ylabel(metric)
		# Add shaded area between 0.9 and 1
		axes[i].axhspan(0.9, 1, color='green', alpha=0.3)
		axes[i].set_ylim(.7, 1.0)

	plt.tight_layout()
	plt.savefig(outpath)  # Save the plot
	plt.show()


# Cross-Validation for K-Fold
def get_metrics_kfold(X, Y, model, k=5):
	kf = KFold(n_splits=k, shuffle=True, random_state=GT_ID)
	accuracies = []
	aucs = []
	precisions = []
	recalls = []
	f1scores = []

	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		Y_pred = model.fit(X_train, Y_train).predict(X_test)
		acc, auc_, precision, recall, f1score = classification_metrics(Y_test, Y_pred)
		accuracies.append(acc)
		aucs.append(auc_)
		precisions.append(precision)
		recalls.append(recall)
		f1scores.append(f1score)

	return np.mean(accuracies), np.mean(auc_), np.mean(precisions), np.mean(recalls), np.mean(f1scores)

# Cross-Validation for Randomized
def get_metrics_randomizedCV(X, Y, model, iterNo=5, test_percent=0.2):
	accuracies = []
	aucs = []
	precisions = []
	recalls = []
	f1scores = []

	for _ in range(iterNo):
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_percent, random_state=GT_ID)

		Y_pred = model.fit(X_train, Y_train).predict(X_test)
		acc, auc_, precision, recall, f1score = classification_metrics(Y_test, Y_pred)
		accuracies.append(acc)
		aucs.append(auc_)
		precisions.append(precision)
		recalls.append(recall)
		f1scores.append(f1score)

	return np.mean(accuracies), np.mean(auc_), np.mean(precisions), np.mean(recalls), np.mean(f1scores)

def plot_metrics(metrics_dict, title):
	classifiers = list(metrics_dict.keys())
	metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-score']

	# Create a figure and axis for the plot
	fig, ax = plt.subplots(figsize=(14, 8))

	# Prepare data for plotting
	for metric in metrics:
		values = [metrics_dict[classifier][metric] for classifier in classifiers]
		ax.plot(classifiers, values, marker='o', label=metric)

	# Add labels, title, and legend
	ax.set_xlabel('Models')
	ax.set_ylabel('Score')
	ax.set_title(f'Classification {title}')
	ax.legend()

	# Show the plot
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.savefig(f'../graphs/CV_{title}.png', format='png', dpi=300)
	plt.close()


def main(dataset,output_file ):
	print("Running cross-validation in cross_validation.py")
	# Pull processed data from etl.py
	X_df, Y_df = etl.get_data(dataset, 0, 0)
	X = np.array(X_df)
	Y = np.array(Y_df)

	# Define models

	models = {
		# "SVM (Linear Kernel)": SVC(kernel='linear', probability=True, random_state=GT_ID),
		"SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=GT_ID),
		"Neural Network": MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', max_iter=1000, random_state=GT_ID),
		"KNN": KNeighborsClassifier(n_neighbors=3),
		"DTBoosted": AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50),
	}
	# Collect metrics for K-Fold CV
	kfold_metrics = {}
	for name, model in models.items():
		print(name)
		print(model)
		acc_k, auc_k, prec_k, rec_k, f1_k = get_metrics_kfold(X, Y, model)
		kfold_metrics[name] = {'Accuracy': acc_k, 'AUC': auc_k, 'Precision': prec_k, 'Recall': rec_k, 'F1-score': f1_k}
		print(f"{name} - K-Fold CV Accuracy: {acc_k}")
		print(f"{name} - K-Fold CV AUC: {auc_k}")
		print(f"{name} - K-Fold CV Precision: {prec_k}")
		print(f"{name} - K-Fold CV Recall: {rec_k}")
		print(f"{name} - K-Fold CV F1-score: {f1_k}")

		with open(output_file, 'a', encoding='utf-8') as file:			
			file.write(f"Name and model: {name} {model}\n")
			file.write("K-fold CV \n")
			file.write(f"Accuracy: {acc_k}\n")
			file.write(f"AUC: {auc_k}\n")
			file.write(f"Precision: {prec_k}\n")
			file.write(f"Recall: {rec_k}\n")
			file.write(f"F1-score: {f1_k}\n")

	# Collect metrics for Randomized CV
	randomized_metrics = {}
	for name, model in models.items():
		acc_r, auc_r, prec_r, rec_r, f1_r = get_metrics_randomizedCV(X, Y, model)
		randomized_metrics[name] = {'Accuracy': acc_r, 'AUC': auc_r, 'Precision': prec_r, 'Recall': rec_r, 'F1-score': f1_r}
		print(f"{name} - Randomized CV Accuracy: {acc_r}")
		print(f"{name} - Randomized CV AUC: {auc_r}")
		print(f"{name} - Randomized CV Precision: {prec_r}")
		print(f"{name} - Randomized CV Recall: {rec_r}")
		print(f"{name} - Randomized CV F1-score: {f1_r}")

		with open(output_file, 'a', encoding='utf-8') as file:			
			file.write(f"Name and model: {name} {model}\n")
			file.write("Randomized CV \n")
			file.write(f"Accuracy: {acc_r}\n")
			file.write(f"AUC: {auc_r}\n")
			file.write(f"Precision: {prec_r}\n")
			file.write(f"Recall: {rec_r}\n")
			file.write(f"F1-score: {f1_r}\n")

	# Plot metrics
	print("Plotting K-Fold CV metrics...")
	plot_metrics(kfold_metrics, f"{dataset} K-Fold CV")

	print("Plotting Randomized CV metrics...")
	plot_metrics(randomized_metrics, f"{dataset} Randomized CV")

if __name__ == "__main__":
	main()
