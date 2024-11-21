
import unittest
import pandas as pd
import numpy as np
import sys
import os
from tabulate import tabulate 

from sklearn.datasets import make_blobs

import text2diag

from config import *


import pandas as pd
import numpy as np
from tabulate import tabulate
from collections import Counter
from scipy.stats import kurtosis, skew


def infer_column_type(series):
    """Infer the column type for the DataFrame."""
    if pd.api.types.is_numeric_dtype(series):
        return 'Numeric'
    elif pd.api.types.is_object_dtype(series):
        return 'Categorical'
    return 'Unknown'

def get_column_info(X, y):
    try:
        column_info_X = []
        column_info_y = []
        
        # Class imbalance check for y
        y_value_counts = Counter(y)
        total_count = sum(y_value_counts.values())
        imbalanced_classes = {k: v / total_count for k, v in y_value_counts.items()}
        # Check for significant imbalance
        threshold = 0.2  # Define your threshold
        imbalanced = {k: v for k, v in imbalanced_classes.items() if v < threshold}
        # Prepare imbalance recommendation
        imbalance_recommendation = ""
        if imbalanced:
            imbalance_recommendation = f"{imbalanced_classes} Class imbalance detected."
        else:
            imbalance_recommendation = f"{imbalanced_classes} No significant class imbalance detected."
        if isinstance(y, pd.Series):
            dtype = y.dtype
            shape = len(y)
            unique_values = y.unique().tolist()
            len_unique_values = len(unique_values)
            example_values = unique_values[:5]
            recommendation = imbalance_recommendation  # Class imbalance recommendation
            column_info_y.append([y.name if y.name else "Target Column", dtype, shape, len_unique_values, example_values, recommendation])


        # Numerical features analysis for X
        numerical_features = X.select_dtypes(include=['int64', 'float64'])
        numerical_recommendations = {}
        
        for column in numerical_features.columns:
            kurt = kurtosis(numerical_features[column])
            skewness = skew(numerical_features[column])
            recommendation = ""
            if abs(skewness) > 1:
                recommendation = f"Consider scaling (high skew) \n{skewness}"
                if skewness > 1:
                    recommendation += " - log transformation recommended"
                else:
                    recommendation += " - normalization recommended"
            elif kurt > 3:
                recommendation = f"Consider scaling (high kurtosis) \n{kurt}"
            else:
                recommendation = f"Scaling not necessary \nk {kurt} skew {skewness}"
            numerical_recommendations[column] = recommendation

        # Categorical features analysis for X
        object_features = X.select_dtypes(include=['object'])
        object_recommendations = {}
        
        for column in object_features.columns:
            unique_vals = object_features[column].nunique()
            value_counts = object_features[column].value_counts(normalize=True)
            top_value_pct = value_counts.iloc[0] * 100
            
            encoding_recommendation = ""
            if unique_vals <= 5:
                encoding_recommendation = f"One-Hot Encoding (few unique values) {unique_vals}"
            elif unique_vals > 5 and top_value_pct > 50:
                encoding_recommendation = f"Frequency Encoding (top category dominates) {unique_vals}"
            elif unique_vals > 5 and top_value_pct < 50:
                encoding_recommendation = f"Target Encoding (distributed categories) {unique_vals}"
            else:
                encoding_recommendation = f"Consider Embedding for High Cardinality {unique_vals}"

            object_recommendations[column] = enconding_recommendation

        # Combine all recommendations into a single list
        all_recommendations = {col: value for col, value in numerical_recommendations.items()}
        # all_recommendations.update({col: += encoding for col, encoding in object_recommendations})
        for col, encoding in object_recommendations:
            all_recommendations[col] = f"\nEncoding Recommendation: {encoding}"

        # Collect column information along with recommendations
        for col in X.columns:
            dtype = infer_column_type(X[col])
            unique_values = X[col].unique().tolist()
            len_unique_values = len(unique_values)
            example_values = [f"{val:.2f}" if isinstance(val, (int, float)) else val for val in unique_values[:3]]
            if pd.api.types.is_numeric_dtype(X[col]):
                min_val = X[col].min()
                max_val = X[col].max()
                median_val = X[col].median()
            else:
                min_val = max_val = median_val = "N/A"  # Set as "N/A" if not numeric
    

            recommendation = all_recommendations[col]
            column_info_X.append([col, dtype, X[col].shape, len_unique_values, example_values, (f"{min_val:.2f},{max_val:.2f},{median_val:.2f}"),recommendation])

        
        

        # Display as a table with the recommendations
        print("\nColumn Headers, Data Types, Len Unique Values, Examples, Stats, and Recommendations for X:")
        print(tabulate(column_info_X, headers=["Column Name", "Data Type","Shape", "Len Unique Values", "Examples","Stats", "Recommendation"], tablefmt="grid"))
        print("Overall X shape: ", X.shape)
       
        # Display as a table with the recommendations
        print("\nColumn Headers, Data Types, Len Unique Values, Examples, and Recommendations for y:")
        print(tabulate(column_info_y, headers=["Column Name", "Data Type","Shape", "Len Unique Values", "Examples", "Recommendation"], tablefmt="grid"))


    except Exception as e:
        print("Error occurred in get_column_info()")
        print(e)


def test_data_etl_input_check(X,y,X_train, X_test, y_train, y_test, 
                        verbose = True):
    if DATA_DEBUG:
        try:
            data_info = [
                    ["X", type(X).__name__, X.shape if hasattr(X, 'shape') else 'N/A'],
                    ["y", type(y).__name__, y.shape if hasattr(y, 'shape') else 'N/A'],
                    ["X_train", type(X_train).__name__, X_train.shape if hasattr(X_train, 'shape') else 'N/A'],
                    ["X_test", type(X_test).__name__, X_test.shape if hasattr(X_test, 'shape') else 'N/A'],
                    ["y_train", type(y_train).__name__, y_train.shape if hasattr(y_train, 'shape') else 'N/A'],
                    ["y_test", type(y_test).__name__, y_test.shape if hasattr(y_test, 'shape') else 'N/A']
                ]
            print(tabulate(data_info, headers=["Variable", "Data Type", "Shape"], tablefmt="grid"))

            get_column_info(X,y)
            # get_column_info(y)
               
            assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Train or Test set for X is empty!"
            assert y_train.shape[0] > 0 and y_test.shape[0] > 0, "Train or Test set for y is empty!"
            assert X_train.shape[1] == X_test.shape[1], "Number of features in train and test sets don't match!"
            assert X_train.shape[0] + X_test.shape[0] == X.shape[0], "Mismatch in total rows between splits and original X!"
            assert y_train.shape[0] + y_test.shape[0] == y.shape[0], "Mismatch in total rows between splits and original y!"
            assert isinstance(X, (pd.DataFrame, np.ndarray)), "X should be a DataFrame or ndarray!"
            assert isinstance(y, (pd.Series, np.ndarray)), "y should be a Series or ndarray!"

            print("All unit tests passed!")
        except Exception as e:
            print(f"An error occurred: {e}")


class TestClusteringFunctions(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset
        self.X, self.y = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=0)

    def test_run_clustering_kmeans(self):
        """Test KMeans clustering functionality."""
        runtime, labels = text2diag.run_clustering('kmeans', 3, random_state=42, X=self.X, y=self.y)
        self.assertEqual(len(labels), len(self.X), "The number of labels should match the number of data points.")
        self.assertIsInstance(runtime, float, "Runtime should be a float.")

    def test_run_clustering_gmm(self):
        """Test GMM clustering functionality."""
        runtime, labels = text2diag.run_clustering('gmm', 3, random_state=42, X=self.X, y=self.y)
        self.assertEqual(len(labels), len(self.X), "The number of labels should match the number of data points.")
        self.assertIsInstance(runtime, float, "Runtime should be a float.")

    def test_run_clustering_invalid_n_clusters(self):
        """Test that ValueError is raised for invalid n_clusters."""
        with self.assertRaises(ValueError) as context:
            text2diag.run_clustering('kmeans', 1, random_state=42, X=self.X, y=self.y)
        self.assertEqual(str(context.exception), "n_clusters must be between 2 and 39.")

    def test_collect_cluster_results(self):
        """Test that results are collected for a valid algorithm."""
        collect_cluster_results(self.X, self.y, 'kmeans')  # This will also print output
        # Check if the pickle file is created (modify path accordingly)
        self.assertTrue(os.path.exists(f'{OUTPUT_DIR_A3}/kmeans_results.pkl'))

    def test_collect_cluster_results_invalid_algorithm(self):
        """Test that ValueError is raised for unsupported algorithms."""
        with self.assertRaises(ValueError) as context:
            collect_cluster_results(self.X, self.y, 'unsupported_algo')
        self.assertEqual(str(context.exception), "Unsupported clustering algorithm. Choose 'kmeans' or 'gmm'.")


if __name__ == '__main__':
    unittest.main()