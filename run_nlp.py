# src/run_nlp.py

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from bayes_opt import BayesianOptimization
from gensim.models import Nmf
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
import seaborn as sns
import matplotlib.pyplot as plt

# Confirm the change
print("Working Directory:", os.getcwd())

# Quick view listdir of working dir
print("\nListdir of working dir:", os.listdir(os.getcwd()))


# Model Definitions
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 75)
        self.fc2 = nn.Linear(75, 19)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class ConvNet(nn.Module):
    def __init__(self, input_size):
        super(ConvNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 289)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.fc2 = nn.Linear(16 * 225, 19)  # Adjust as needed

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, int(289**0.5), int(289**0.5))  # reshape for convolution
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc2(x))
        return x


class LSTM(nn.Module):
    def __init__(self, input_size):
        super(LSTM, self).__init__()
        self.fc1 = nn.Linear(input_size, 289)
        self.lstm = nn.LSTM(289, 300)
        self.fc2 = nn.Linear(300, 19)

    def forward(self, x, h0, c0):
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), 1, -1)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = torch.sigmoid(self.fc2(out))
        return out, h0, c0


class BiLSTM(nn.Module):
    def __init__(self, input_size):
        super(BiLSTM, self).__init__()
        self.fc1 = nn.Linear(input_size, 289)
        self.lstm = nn.LSTM(289, 300, bidirectional=True)
        self.fc2 = nn.Linear(300, 19)

    def forward(self, x, h0, c0):
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), 1, -1)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = torch.sigmoid(self.fc2(out))
        return out, h0, c0


class ConvLSTM(nn.Module):
    def __init__(self, input_size):
        super(ConvLSTM, self).__init__()
        self.fc1 = nn.Linear(input_size, 289)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.fc2 = nn.Linear(16 * 225, 289)
        self.lstm = nn.LSTM(289, 300)
        self.fc3 = nn.Linear(300, 19)

    def forward(self, x, h0, c0):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, int(289**0.5), int(289**0.5))
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc2(x))
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = torch.sigmoid(self.fc3(out))
        return out, h0, c0


# Data Processing
def csv_load_helper(x):
    return np.array([float(e) for e in x.replace('[', '').replace(']', '').replace('\n', '').split()])


def y_to_onehot(y):
    meep = np.eye(19, dtype='uint8')[y - 1].sum(axis=0)
    return meep


# Load data
def load_data():
    with open(f'data/doc2vec_dataset_full.pkl', 'rb') as f:
        df_X = pickle.load(f)
    df_X['processed_text'] = df_X['processed_text'].apply(csv_load_helper)
    X = np.stack(df_X['processed_text'].to_numpy())
    y = np.stack(df_X['DIAG_GROUPS_OF_FIRST_HADM_ONLY'].apply(np.array).apply(y_to_onehot).to_numpy())
    return X, y

def assess_raw_data(X, y):
    """
    Assess basic statistics and properties about the raw data (X, y).
    
    Args:
        X: Input feature data (numpy array or pandas DataFrame).
        y: Target labels (numpy array or pandas Series).
    
    Returns:
        None (prints statistics directly).
    """
    # Check if X and y are numpy arrays or pandas dataframes/series
    if isinstance(X, np.ndarray):
        print("X is a numpy array")
    elif isinstance(X, pd.DataFrame):
        print("X is a pandas DataFrame")
    else:
        print("Unknown type for X")
    
    if isinstance(y, np.ndarray):
        print("y is a numpy array")
    elif isinstance(y, pd.Series):
        print("y is a pandas Series")
    else:
        print("Unknown type for y")
    
    print("\nShape and size of X and y:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Statistics for X (features)
    print("\nStatistics for X (features):")
    if isinstance(X, np.ndarray):
        print(f"Min length of elements in X: {min([len(x) for x in X])}")
        print(f"Max length of elements in X: {max([len(x) for x in X])}")
        print(f"Median length of elements in X: {np.median([len(x) for x in X])}")
    elif isinstance(X, pd.DataFrame):
        # For DataFrame, we can check basic stats of each column
        print("Columns in X:")
        print(X.describe())
    
    # Statistics for y (targets)
    print("\nStatistics for y (targets):")
    if isinstance(y, np.ndarray):
        print(f"Min value in y: {np.min(y)}")
        print(f"Max value in y: {np.max(y)}")
        print(f"Median value in y: {np.median(y)}")
        print(f"Unique values in y: {np.unique(y)}")
    elif isinstance(y, pd.Series):
        # For Series, basic stats
        print("y basic stats:")
        print(y.describe())

    # Data Types
    print("\nData Types:")
    if isinstance(X, np.ndarray):
        print(f"X data type: {X.dtype}")
    elif isinstance(X, pd.DataFrame):
        print(f"X data types: \n{X.dtypes}")
    
    print(f"y data type: {y.dtype}")
    
    # Check for NaN or missing values
    print("\nMissing Values:")
    if isinstance(X, np.ndarray):
        print(f"Missing values in X: {np.isnan(X).sum()}")
    elif isinstance(X, pd.DataFrame):
        print(f"Missing values in X: \n{X.isna().sum()}")
    
    if isinstance(y, np.ndarray):
        print(f"Missing values in y: {np.isnan(y).sum()}")
    elif isinstance(y, pd.Series):
        print(f"Missing values in y: {y.isna().sum()}")

    # Plot correlation heatmap for X
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(X, rowvar=False)
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", square=True, cbar=True)
    plt.title("Correlation Heatmap of Features in X")
    plt.show()

    # Plot class distribution for each of the 19 labels in multi-label `y`
    plt.figure(figsize=(10, 6))

    # Calculate the ratio for each label (column) by summing each column and dividing by the number of samples
    label_counts = np.sum(y, axis=0)
    label_ratios = label_counts / y.shape[0]

    # Create a bar plot
    sns.barplot(x=np.arange(y.shape[1]), y=label_ratios)
    plt.xlabel("Class Label")
    plt.ylabel("Ratio")
    plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
    plt.title("Class Distribution Ratio for Multi-Label y")
    plt.xticks(ticks=np.arange(y.shape[1]), labels=[f'Label {i}' for i in range(y.shape[1])], rotation=45)
    plt.show()
        
    print("\n------------------------------------------\n")


def get_device():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = torch.device('cuda')
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device('cpu')
    return device


# Training loop
def eval_model(lr, model, X, y, epochs=8, device=None):
    """
    Evaluate the model with given learning rate on the dataset X and y.
    This function supports CUDA if device is passed.
    """
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Move model to device (GPU or CPU)
    model = model.to(device)
    
    # Prepare data for training
    X_train_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    # DataLoader for batching
    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    loader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    
    # Track the best performance
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in loader_train:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)  # Accumulate loss

        # Calculate average loss
        train_loss /= len(loader_train.dataset)
        
        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_train_tensor)
            test_loss = criterion(test_outputs, y_train_tensor)
            
            # Convert outputs and labels to CPU and NumPy for metric calculations
            # test_outputs_np = test_outputs.sigmoid().cpu().numpy()  # Sigmoid for multi-label probability
            test_outputs_np = test_outputs.cpu().numpy()  # Sigmoid for multi-label probability
            y_train_np = y_train_tensor.cpu().numpy()
            
            # Binarize predictions with a threshold of 0.5
            # predicted_labels = (test_outputs_np >= 0.55).astype(int)
            predicted_labels = test_outputs_np.astype(int)

            # Debug: Print predicted vs actual values for the first 50 instances
            # print("\nPredicted vs. Actual for first 50 instances:")
            # for i in range(50):
            #     print(f"Instance {i + 1}")
            #     print(f"test_outputs_np: {test_outputs_np[i]}")
            #     print(f"Predicted: {predicted_labels[i]}")
            #     print(f"Actual   : {y_train_np[i]}")
            #     print("-" * 30)
            
            # Calculate accuracy, AUC-ROC, and F1-score for multi-label classification
            # Individual label accuracy (mean accuracy for each label)
            label_accuracy = (predicted_labels == y_train_np).mean(axis=0)
            avg_label_accuracy = label_accuracy.mean()  # Average accuracy across all labels

            # Calculate macro AUC-ROC
            try:
                auc_roc = roc_auc_score(y_train_np, test_outputs_np, average="macro")
            except ValueError:
                auc_roc = float("nan")  # Handle cases where AUC-ROC can't be calculated

            # Calculate macro F1-score
            f1 = f1_score(y_train_np, predicted_labels, average="macro")

        # Save the model if it achieves a new best loss
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model.state_dict()
            torch.save(best_model, f'best_model_{type(model).__name__}.pt')

        # Print the metrics
        print(f"Epoch {epoch+1}/{epochs}, "
            f"Test Loss: {test_loss.item():.4f}, "
            f"Avg Label Accuracy: {avg_label_accuracy:.4f}, "
            f"AUC-ROC: {auc_roc:.4f}, "
            f"F1-Score: {f1:.4f}")
    
    return -best_loss.item()


# Topic Modeling
def generate_nmf_model():
    fcrnn = pd.read_csv('filtered_cleaned_raw_nursing_notes_processed.csv')
    fcrnn['processed_text'] = fcrnn['processed_text'].apply(eval)
    texts_dict = Dictionary(fcrnn['processed_text'])
    corpus = [texts_dict.doc2bow(text) for text in fcrnn['processed_text']]
    nmf = Nmf(corpus, num_topics=100, id2word=texts_dict, passes=10)
    temp_file = datapath("nmf_bow_model")
    nmf.save(temp_file)


# Main script to run the models with Bayesian Optimization
if __name__ == "__main__":
    # Get the device (GPU if available, otherwise CPU)
    device = get_device()
    
    # Load data
    X, y = load_data()

    # assess_raw_data(X, y)
    
    # Initialize models
    models = [MLP(input_size=500), ConvNet(input_size=500), LSTM(input_size=500),
              BiLSTM(input_size=500), ConvLSTM(input_size=500)]
    
    # Run Bayesian Optimization for each model
    for model in models:
        bounds = {'lr': (0.0001, 0.1)}  # Define hyperparameter space
        optimizer = BayesianOptimization(f=lambda lr: eval_model(lr, model, X, y, device=device), pbounds=bounds)
        optimizer.maximize(init_points=10, n_iter=30)
        print(f'Best model for {type(model).__name__}:', optimizer.max)
