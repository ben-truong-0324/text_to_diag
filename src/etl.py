
import utils
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import unittest

from config import *

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def calculate_index_date(events, mortality, deliverables_path):
    return None

def filter_events(events, indx_date, deliverables_path):
    return None

def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    return None, None

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    pass

def calc_speed_accel(df):
    df['speed'] = 0.0
    df['acceleration'] = 0.0
    df['time'] = pd.to_datetime(df['time'])

    # Haversine function for distance between two lat-long points
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371e3  # Earth radius in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c  # distance in meters

    for track_id, group in df.groupby('track_id'):

        for i in range(1, len(group)):
            distance = haversine(group.iloc[i-1]['latitude'], group.iloc[i-1]['longitude'],
                                 group.iloc[i]['latitude'], group.iloc[i]['longitude'])
            time_diff = (group.iloc[i]['time'] - group.iloc[i-1]['time']).total_seconds()
            if time_diff > 0:
                df.loc[group.index[i], 'speed'] = distance / time_diff
                df.loc[group.index[i], 'acceleration'] = (df.loc[group.index[i], 'speed'] - 
                                                          df.loc[group.index[i-1], 'speed']) / time_diff
                
    return df


def calc_features(df):
    # Ensure the timestamp is in datetime format
    df['time'] = pd.to_datetime(df['time'])

    # Extract 'hour of day' and 'day of week'
    df['hour_of_day'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek

    stop_threshold = 0.5

    # Create empty lists to store the running results
    running_stop_sums = []
    running_avg_stop_lengths = []

    # Iterate over each track_id
    for track_id, group in df.groupby('track_id'):
        group = group.sort_values(by='time')  # Sort by timestamp within each track_id
        
        # Identify stops: a stop is where speed is less than the threshold
        group['is_stop'] = group['speed'] < stop_threshold
        
        # Initialize the running sum for number of stops and cumulative stop duration
        running_sum = 0
        cumulative_stop_duration = 0
        
        # Track the stop change (transitions between moving and stopped)
        group['stop_change'] = group['is_stop'].astype(int).diff().fillna(0)
        stop_start_time = None 
        
        # Track cumulative stop count and running average stop length as we go through each row
        for i in range(len(group)):
            # Check if there is a transition from not stopped to stopped
            if group.iloc[i]['stop_change'] == 1:
                running_sum += 1
                stop_start_time = group.iloc[i]['time']
                
            # If the current row is part of a stop, calculate stop duration if the stop ends
            if stop_start_time is not None and group.iloc[i]['stop_change'] == -1:  # End of stop

                stop_end_time = group.iloc[i]['time']
                stop_duration = (stop_end_time - stop_start_time).total_seconds()
                cumulative_stop_duration += stop_duration
                stop_start_time = None
            
            # Calculate the running average stop length (avoid division by zero)
            running_avg = cumulative_stop_duration / running_sum if running_sum > 0 else 0
            
            # Append the running stop sum and running average stop length at the current index
            running_stop_sums.append(running_sum)
            running_avg_stop_lengths.append(running_avg)

    # Append the running sum of stops and running average stop length to the original DataFrame
    df['running_num_stops'] = running_stop_sums
    df['running_avg_length_of_stops'] = running_avg_stop_lengths

    return df

def get_data(dataset, do_scaling, do_pca):
    if "credit" in dataset:
        df = pd.read_csv(CREDIT_DATA_PATH, header=None)
        X_df = df.iloc[:, :-1]  # All rows, all columns except the last one
        Y_df = df.iloc[:, -1]
        le = LabelEncoder()
        Y_df = le.fit_transform(Y_df)
        Y_df = 1 - Y_df
        label_encoders = {}
        for col in X_df.columns:
            if X_df[col].dtype == object:
                # Create and fit a LabelEncoder for each categorical column
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col])
                label_encoders[col] = le
    elif "gps" in dataset:
        print("loading gps dataset")
        # data = np.load(file_path)
        df1 = pd.read_csv(GPS_DATA_PATH1)
        df2 = pd.read_csv(GPS_DATA_PATH2)
        df_merged = df2.merge(df1[['id', 'car_or_bus']], left_on='track_id', right_on='id', how='left')
        df_merged = calc_speed_accel(df_merged) #get speed and acceleration
        df_merged = calc_features(df_merged)

        #get hour of day, day of week
        #get number of stops
        #get avg length of stops
        unique_track_ids = df_merged['track_id'].unique()

        X_df = df_merged[[
            #'latitude', 'longitude',
        'speed','acceleration',
        'hour_of_day', 
        'day_of_week',
        'running_num_stops', 'running_avg_length_of_stops',
        ]]
        Y_df = df_merged[[
            # 'track_id',
            'car_or_bus']]
        
        Y_df = Y_df - 1 #switch from 1/2 to 1/0
        Y_df = Y_df.values.ravel()
        # Convert both to NumPy arrays
        X_df = X_df.to_numpy() if isinstance(X_df, pd.DataFrame) else X_df
        # Y_df = Y_df.to_numpy() if isinstance(Y_df, pd.DataFrame) else Y_df
        
    else: 
        raise ValueError("Invalid dataset specified. Check config.py")

   
    if do_scaling:
        scaling=StandardScaler()
        scaling.fit(X_df) 
        # X_df=scaling.transform(X_df)
        X_df_scaled = scaling.transform(X_df)  # Scaled data is a NumPy array
        X_df = pd.DataFrame(X_df_scaled, )
        print("Scaling implemneted")
    if do_pca:    
        pca = PCA(n_components=.95)  # get enough for 95% of var explained
        pca.fit(X_df)
        # X_df = pca.fit_transform(X_df)
        X_df_pca = pca.fit_transform(X_df)  # PCA transforms into NumPy array
        X_df = pd.DataFrame(X_df_pca, index=X_df.index, 
                        columns=[f'PC{i+1}' for i in range(X_df_pca.shape[1])])
        print("PCA implemneted")

    # print("Extracted Y_df:")
    # print(Y_df)
    # print("X_df:")
    # print(X_df)
    print(type(X_df))
    print(type(Y_df))

    return X_df, Y_df

def main():

    print("hello")

    data1_path = '../data/bodmas_dataset/bodmas.npz'
    data1 = np.load(data1_path)
    print("loaded")
    print(data1)
    X = data1['X']
    Y = data1['y']

    print("trying to do some correlation")
    # Convert to pandas DataFrame
    X_df = pd.DataFrame(X)
    Y_df = pd.Series(Y)
    # Assuming Y_df is a Series and needs to be added to X_df for correlation analysis


    print("printing X_df")
    print(X_df)


    # Scale data before applying PCA
    scaling=StandardScaler()

    # Use fit and transform method 
    scaling.fit(X_df)
    X_df=scaling.transform(X_df)

    # Apply PCA
    print("fitting for PCA")
    print("attributes + target of size ",X_df.shape)
    pca = PCA(n_components=.95)  # Adjust number of components as needed
    pca.fit(X_df)
    
    print("reduced X:")
    X_reduced = pca.fit_transform(X_df)
    print(X_reduced.shape)

   
    # Cumulative explained variance
    cum_variance = np.cumsum(pca.explained_variance_ratio_)
    # Plot cumulative explained variance
    plt.plot(cum_variance)

    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.grid(True)
    plt.show()
    # Find n_components that explain at least 95% of the variance
    n_components_95 = np.argmax(cum_variance >= 0.95) + 1
    print(f"Number of components explaining 95% of variance: {n_components_95}")

   

if __name__ == "__main__":
    main()