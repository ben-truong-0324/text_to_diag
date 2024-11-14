
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import etl  

# Function to compute metrics
def classification_metrics(Y_true, Y_pred):
    acc = accuracy_score(Y_true, Y_pred)
    auc_ = roc_auc_score(Y_true, Y_pred) if len(set(Y_true)) > 1 else 'N/A'
    precision = precision_score(Y_true, Y_pred, average='weighted')
    recall = recall_score(Y_true, Y_pred, average='weighted')
    f1score = f1_score(Y_true, Y_pred, average='weighted')
    return acc, auc_, precision, recall, f1score

# Track training and validation 
def get_training_and_cv_losses(X, Y, model, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_losses = []
    cv_losses = []
    accuracies = []
    aucs = []
    precisions = []
    recalls = []
    f1scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Fit model and record training loss
        model.fit(X_train, Y_train)
        train_loss = model.loss_curve_[-1]  # Last value in the loss curve
        train_losses.append(train_loss)

        # Predict and compute validation loss
        Y_pred = model.predict(X_test)
        le = LabelEncoder()
        Y_test = pd.Series(le.fit_transform(Y_test))
        Y_pred = pd.Series(le.fit_transform(Y_pred))
        cv_loss = np.mean(Y_pred != Y_test)  # Classification error
        cv_losses.append(cv_loss)
        
        # Compute accuracy
        acc, auc, precision, recall, f1score= classification_metrics(Y_test, Y_pred)
        accuracies.append(acc)
        aucs.append(auc)
        precisions.append(precision)
        recalls.append(recall)
        f1scores.append(f1score)

    return train_losses, cv_losses, accuracies, aucs, precisions, recalls, f1scores

# Cross-Validation for Randomized
def get_training_and_cv_losses_randomized(X, Y, model, iterNo=5, test_percent=0.2):
    train_losses = []
    cv_losses = []
    accuracies = []
    aucs = []
    precisions = []
    recalls = []
    f1scores = []
    for _ in range(iterNo):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_percent, random_state=42)

        # Fit model and record training loss
        model.fit(X_train, Y_train)
        train_loss = model.loss_curve_[-1]  # Last value in the loss curve
        train_losses.append(train_loss)

        # Predict and compute validation loss
        Y_pred = model.predict(X_test)
        cv_loss = np.mean(Y_pred != Y_test)  # Classification error
        cv_losses.append(cv_loss)
        
        # Compute metrics
        acc, auc, precision, recall, f1score= classification_metrics(Y_test, Y_pred)
        accuracies.append(acc)
        aucs.append(auc)
        precisions.append(precision)
        recalls.append(recall)
        f1scores.append(f1score)

    return train_losses, cv_losses, accuracies, aucs, precisions, recalls, f1scores

def plot_loss_and_error(train_losses, cv_losses, model_name):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, cv_losses, label='Cross-Validation Loss')
    plt.xlabel('Fold/Iteration')
    plt.ylabel('Loss/Error')
    plt.title(f'{model_name} - Training and CV Loss/Error')
    plt.legend()
    plt.show()

def main():
    print("Running loss curve in loss_curve.py")
    # Pull processed data from etl.py
    # path_bodmas_dataset = '../data/bodmas_dataset/bodmas.npz'
    # X_df, Y_df = etl.get_data(path_bodmas_dataset, 0, 0)
    path_credit_dataset = '../data/credit+approval/crx.data'
    X_df, Y_df = etl.get_data(path_credit_dataset, 0, 0)
    X = np.array(X_df)
    Y = np.array(Y_df)    

    models = {
        "SVM (Linear Kernel)": SVC(kernel='linear', probability=True, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3)
    }

    # Collect and plot training and CV losses for Neural Network
    for name, model in models.items():
        if name == "Neural Network":
            print(f"Calculating losses for {name}...")
            train_losses, cv_losses, accuracies, aucs, precisions, recalls, f1scores = get_training_and_cv_losses(X, Y, model)
            plot_loss_and_error(train_losses, cv_losses, name)

if __name__ == "__main__":
    main()
