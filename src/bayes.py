import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import zscore
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from bayes_opt import BayesianOptimization
import time
from models import *
models = [
    MLP(input_size=500),
    ConvNet(input_size=500),
    LSTM(input_size=500),
    BiLSTM(input_size=500),
    ConvLSTM(input_size=500)
]
for model_name in FARSIGHT_MODELS:
    
# BATCH_SIZE always 128, train for 8 epochs
for model in models:
  def eval_model(lr, model=model, X = X, y = y, epochs = 8):
    splitter = ShuffleSplit(n_splits=5, test_size=0.2) # 5 fold CV as specified in paper
    for train, test in splitter.split(X, y):
      X_train, X_test = X[train], X[test]
      y_train, y_test = y[train], y[test]
      optimizer = optim.Adam(model.parameters(), lr=lr)
      criterion = nn.BCEWithLogitsLoss()
      epochs_needed = []
      X_train = torch.tensor(X_train, dtype=torch.float32)
      y_train = torch.tensor(y_train, dtype=torch.float32)
      X_test = torch.tensor(X_test, dtype=torch.float32)
      y_test = torch.tensor(y_test, dtype=torch.float32)
      dataset_train = TensorDataset(X_train, y_train)
      patience = 5
      loader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
      best_val_acc = 0.0
      train_losses, train_accuracies = [], []
      test_losses, test_accuracies = [], []

      best_loss = float('inf')
      # epochs
      for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        # train for batches
        for X_batch, y_batch in loader_train:
          optimizer.zero_grad()
          outputs = model(X_batch)
          loss = criterion(outputs, y_batch)
          loss.backward()
          optimizer.step()
        model.eval()
        with torch.no_grad():
          train_outputs = model(X_train)
          test_outputs = model(X_test)
          train_loss = criterion(train_outputs, y_train)
          test_loss = criterion(test_outputs, y_test)
          train_losses.append(train_loss.item())
          test_losses.append(test_loss.item())
        if test_loss < best_loss:
          best_loss = test_loss
          best_model = model.state_dict()
          torch.save(best_model, f'best_doc2Vec_{model}.pt')


        # Evaluate the model on the test set


          print(f"Model: {type(model).__name__}, lr: {lr}, epochs: {epoch}, Avg Test Loss: {np.mean(test_losses)}")
          # Print evaluation metrics
        with torch.no_grad():
            outputs = model(X_test)
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
        #print(f"Avg Train Loss: {np.mean(train_losses)}, Avg Test Loss: {np.mean(test_losses)}")
        #print(f"Test Accuracy: {accuracy}")
    return -np.mean(test_losses)
  # bayesian optimization on all 5 models....this might take a fews days? :3


  bounds = {
    'lr': (0.0001, 0.1)
  }
  optimizer = BayesianOptimization(
      f=eval_model,
      pbounds=bounds,
      verbose=2,  # verbose = 1 prints only when a maximum
      # is observed, verbose = 0 is silent
      random_state=1,
  )
  optimizer.maximize(
      init_points=10,
      n_iter=30
  )
  print(f'================= BEST MODEL for {model}', optimizer.max, '=================')