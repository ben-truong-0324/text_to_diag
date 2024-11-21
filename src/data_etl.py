import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import time
from config import *
import data_plots
import pickle



def get_pd_data(dataset, do_floor_cap, floor_cap_per):
    pandas_df = pd.read_csv('../data/sp500_data_sparkdftocsv/part-00000-5d1b3de8-0339-48a0-ac2b-a0f474691b5b-c000.csv')
    if do_floor_cap:
        start_time = time.time()
        print(f"applying floor at {floor_cap_per}, cap at {1-floor_cap_per}")
        # Loop through each feature to apply capping and flooring
        features_to_skip = ["bollinger_scaled","rsi_scaled",]
        for feature in PYSPARK_FEATURES:
            if feature in features_to_skip:
                continue
            upper_cap = pandas_df[feature].quantile(1-floor_cap_per)  # Calculate upper cap
            lower_floor = pandas_df[feature].quantile(floor_cap_per)  # Calculate lower floor
            # Apply capping
            pandas_df[feature] = np.where(pandas_df[feature] > upper_cap, upper_cap, pandas_df[feature])
            # Apply flooring
            pandas_df[feature] = np.where(pandas_df[feature] < lower_floor, lower_floor, pandas_df[feature])
        end_time = time.time()
        print(("Time to apply capping/flooring: " + str(end_time - start_time) + "s"))
    return pandas_df

def get_sp500_data(dataset, do_scaling, do_pca, do_panda):
    if "sp500" in DATASET_SELECTION:
        # Load the data into a Pandas DataFrame
        sp500_df = dataset
        sp500_df.columns = sp500_df.iloc[0]
        sp500_df = sp500_df[1:]
        sp500_df = sp500_df.dropna(subset=['Close'])
        sp500_df['Close'] = pd.to_numeric(sp500_df['Close'], errors='coerce')
        sp500_df['Volume'] = pd.to_numeric(sp500_df['Volume'], errors='coerce')
        
        # Calculate Moving Averages
        sp500_df['MA_10d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=10).mean()) / x)
        sp500_df['MA_25d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=25).mean()) / x)
        sp500_df['MA_50d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=50).mean()) / x)
        sp500_df['MA_200d'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: (x - x.rolling(window=200).mean()) / x)

        tolerance = 1e-5
        sp500_df['MA_vol_10d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=10).mean()) / (x.rolling(window=10).mean() + tolerance))
        sp500_df['MA_vol_25d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=25).mean()) / (x.rolling(window=10).mean() + tolerance))
        sp500_df['MA_vol_50d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=50).mean()) / (x.rolling(window=10).mean() + tolerance))
        sp500_df['MA_vol_200d'] = sp500_df.groupby('Ticker')['Volume'].transform(lambda x: (x - x.rolling(window=200).mean()) / (x.rolling(window=10).mean() + tolerance))
        
        # Calculate MACD
        def calculate_macd(df, short_window, long_window, signal_window, tolerance=1e-4):
            short_ema = sp500_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=short_window, adjust=False).mean())
            long_ema = sp500_df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=long_window, adjust=False).mean())
            macd = short_ema - long_ema
            signal_line = macd.ewm(span=signal_window, adjust=False).mean()
            macd = macd.clip(lower=tolerance) 
            return (macd - signal_line)/macd
        # sp500_df['macd'], sp500_df['signal_line'] = calculate_macd(sp500_df, short_window, long_window, signal_window)
        sp500_df['macd_signal_ratio'] = calculate_macd(sp500_df, short_window, long_window, signal_window)

        
        # Calculate Bollinger Bands

        def calculate_bollinger_ratio(df, window_size, tolerance=1e-8):
            bollinger_mid = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=window_size).mean())
            bollinger_std = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=window_size).std())
            bollinger_std = bollinger_std.clip(lower=tolerance)  # Clip the standard deviation to a minimum value
            return (df['Close'] - bollinger_mid) / (bollinger_std * 2)

        sp500_df['bollinger_band_ratio'] = calculate_bollinger_ratio(sp500_df, window_size)
       
        # Calculate RSI
        def calculate_rsi(df, period):
            
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            exponentially_weighted_moving_average_gain = gain.ewm(com=period - 1, adjust=False).mean()
            exponentially_weighted_moving_average_loss = loss.ewm(com=period - 1, adjust=False).mean()
            rs = exponentially_weighted_moving_average_gain / exponentially_weighted_moving_average_loss
            return 100 - (100 / (1 + rs))
        sp500_df['rsi'] = calculate_rsi(sp500_df, rsi_period)
       

        # Calculate ROC
        sp500_df['roc'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: ((x - x.shift(roc_period)) /  (x.shift(roc_period) + tolerance) ).clip(lower=tolerance)  * 100)

        # Calculate DoD Delta
        sp500_df['dod_delta'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: np.where(x < x.shift(-1), 1, 0))
        sp500_df['dod5_delta'] = sp500_df.groupby('Ticker')['Close'].transform(lambda x: np.where(x < x.shift(-5), 1, 0))

        # Drop NaN values
        sp500_df.dropna(inplace=True)
        columns_to_drop = ['Open', 'Close', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Ticker']
        sp500_df.drop(columns=columns_to_drop, inplace=True)
        print("Columns:", sp500_df.columns)
        print("Shape:", sp500_df.shape)
        print(sp500_df.head(20))

        with open(SP500_PROCESSED_DATA_PATH, 'wb') as f:
            pickle.dump(sp500_df, f)
        print(f"Results saved to {SP500_PROCESSED_DATA_PATH}")

        def print_df_stats(df):
            print("Shape:", df.shape)
            print("Columns:", df.columns)
            print("Data Types:", df.dtypes)

            for col in df.columns:
                print(f"\nColumn: {col}")
                print("Unique Values:", df[col].nunique())
                if pd.api.types.is_numeric_dtype(df[col]):
                    print("Min:", df[col].min())
                    print("Max:", df[col].max())
                    print("Median:", df[col].median())
                    print("Example Values:", df[col].sample(3).tolist())
                else:
                    print("Example Values:", df[col].sample(3).tolist())

        

        print_df_stats(sp500_df)



    else: 
        raise ValueError("Invalid dataset specified. Check config.py")
          
    
    #probably need to deprecate
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
        X_df_pca = pca.fit_transform(X_df)  # PCA transforms into NumPy array
        X_df = pd.DataFrame(X_df_pca, index=X_df.index, 
                        columns=[f'PC{i+1}' for i in range(X_df_pca.shape[1])])
        print("PCA implemneted")
    if do_panda:
        pass

    return sp500_df


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
# Data Processing
def csv_load_helper(x):
    return np.array([float(e) for e in x.replace('[', '').replace(']', '').replace('\n', '').split()])


def y_to_onehot(y):
    # meep = np.eye(19, dtype='uint8')[y - 1].sum(axis=0)
    # return meep
    onehot = np.eye(19, dtype='uint8')[y - 1].sum(axis=0)
    return onehot.astype(np.float32)

def get_data(dataset, do_scaling, do_log_transform):
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
        # data = np.load(file_path)
        df1 = pd.read_csv(GPS_DATA_PATH1)
        df2 = pd.read_csv(GPS_DATA_PATH2)
        df_merged = df2.merge(df1[['id', 'car_or_bus']], left_on='track_id', right_on='id', how='left')
        df_merged = calc_speed_accel(df_merged) #get speed and acceleration
        df_merged = calc_features(df_merged)

        #get hour of day, day of week
        #get number of stops
        #get avg length of stops
        df_merged['track_id'].unique()

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
    elif 'phishing' in dataset:
        df = pd.read_csv(PHISHING_DATA_PATH, header=None)
        df.columns = df.iloc[0]  # Make the first row the header
        df = df.drop(0).reset_index(drop=True)
        # print(df.head())
        X_df = df.iloc[:, 1:-1]  # All rows, all columns except the last one
        Y_df = df.iloc[:, -1]

        # Columns that should remain as float
        float_columns = ['PctExtHyperlinks', 'PctExtResourceUrls', 'PctNullSelfRedirectHyperlinks']

        # Convert all columns in X_df to int, except the specified float columns
        for col in X_df.columns:
            if col in float_columns:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')  # Cast as float
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce').astype(int)  # Cast as int

        # Convert Y_df to int
        Y_df = pd.to_numeric(Y_df, errors='coerce').astype(int)
    elif 'bd4h' in dataset:
        with open(BD4H_DATA_PATH, 'rb') as f:
            df_X = pickle.load(f)
        df_X['processed_text'] = df_X['processed_text'].apply(csv_load_helper)
        X_df = np.stack(df_X['processed_text'].to_numpy())
        Y_df = np.stack(df_X['DIAG_GROUPS_OF_FIRST_HADM_ONLY'].apply(np.array).apply(y_to_onehot).to_numpy())
    elif 'NMF_BOW' in dataset:
        with open(NMF_BOW_DATA_PATH, 'rb') as f:
            df_X = pickle.load(f)
        print(df_X.shape)
        df_X['processed_text'] = df_X['processed_text'].apply(csv_load_helper)
        X_df = np.stack(df_X['processed_text'].to_numpy())
        Y_df = np.stack(df_X['DIAG_GROUPS_OF_FIRST_HADM_ONLY'].apply(np.array).apply(y_to_onehot).to_numpy())
    
    elif 'sp500' in dataset:
        if not os.path.exists(SP500_PROCESSED_DATA_PATH): 
            dataset = pd.read_csv(SP500_DATA_PATH, header=None)
            sp500_df = get_sp500_data(dataset, 0, 0, 0)
        else:
            with open(SP500_PROCESSED_DATA_PATH, 'rb') as f:
                sp500_df = pickle.load(f)
        
            sp500_df.rename(columns={sp500_df.columns[0]: 'temp_nan'}, inplace=True)
            sp500_df.drop('temp_nan', axis=1, inplace=True)
            X_df = sp500_df.iloc[:, :-2]  # All columns, all columns except the last 2
            if pred_for_5d_delta:
                Y_df = sp500_df.loc[:, 'dod5_delta']
            else:
                Y_df = sp500_df.loc[:, 'dod_delta']
            print(X_df.columns)
    else: 
        print("#"*18)
        raise ValueError("Invalid dataset specified. Check config.py")

    if do_log_transform:    
        if 'phishing' in dataset:
            columns_to_log_transform = []
            for col in X_df.columns:
                if pd.api.types.is_numeric_dtype(X_df[col]):  # Ensure the column is numeric
                    col_min = X_df[col].min()
                    col_max = X_df[col].max()
                    col_median = X_df[col].median()
                    
                    # Check if (max - min) is more than 3 times (median - min) and only for those that have mulitple unique values to avoid log transform bool
                    if (col_max - col_min) > 3 * (col_median - col_min) and X_df[col].nunique() > 4:
                        columns_to_log_transform.append(col)

            # Apply log transformation to the selected columns
            for col in columns_to_log_transform:
                # Adding 1 to avoid issues with zero or negative values
                X_df[col] = np.log1p(X_df[col])
        elif 'sp500' in dataset:
            X_df['roc'] = np.log(X_df['roc'])
        else:
            pass
   
    if do_scaling:
        if 'credit' in dataset:
            scaling=StandardScaler()
            scaling.fit(X_df) 
            X_df_scaled = scaling.transform(X_df)  # now is NumPy array
            X_df = pd.DataFrame(X_df_scaled, ) #now pd df
        elif 'phishing' in dataset:
            # Initialize the scaler
            scaler = StandardScaler()  # or MinMaxScaler()
            columns_to_scale = [col for col in X_df.columns if X_df[col].nunique() > 4]
            X_df[columns_to_scale] = scaler.fit_transform(X_df[columns_to_scale])
        elif 'sp500' in dataset:
            columns_to_clip_and_scale = ['MA_10d', 'MA_25d', 'MA_50d', 'MA_200d', 'MA_vol_10d', 'MA_vol_25d', 'MA_vol_50d', 'MA_vol_200d', 'macd_signal_ratio', 'roc']

            X_df[columns_to_clip_and_scale] = X_df[columns_to_clip_and_scale].clip(-2, 2)
            scaler = MinMaxScaler(feature_range=(-2, 2))
            X_df[columns_to_clip_and_scale] = scaler.fit_transform(X_df[columns_to_clip_and_scale])


    

    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)  # Convert to DataFrame
    if Y_df.ndim == 1:
        # If it's 1D, convert to Pandas Series
        Y_df = pd.Series(Y_df)
    else:
        # If it's 2D, convert to Pandas DataFrame
        Y_df = pd.DataFrame(Y_df)
    return X_df, Y_df

def graph_raw_data(X_df, Y_df):
    raw_data_outpath = f'{OUTPUT_DIR_RAW_DATA_A3}/ver{DRAFT_VER_A3}/raw_data_assessment'
    os.makedirs(raw_data_outpath, exist_ok=True)
    # Check if Y_df is multi-label (2D) or single-label (1D)
    if Y_df.ndim == 1:  # Single-label
        if not os.path.exists(f'{raw_data_outpath}/feature_heatmap.png'):
            # Plot class imbalance, feature violin, heatmap, etc.
            data_plots.graph_class_imbalance(Y_df, 
                                             f'{raw_data_outpath}/class_imbalance.png')
            data_plots.graph_feature_violin(X_df, Y_df, 
                                             f'{raw_data_outpath}/feature_violin.png')
            data_plots.graph_feature_heatmap(X_df, Y_df,
                                             f'{raw_data_outpath}/feature_heatmap.png')
            data_plots.graph_feature_histogram(X_df, 
                                             f'{raw_data_outpath}/feature_histogram.png')
            data_plots.graph_feature_correlation(X_df, Y_df,
                                             f'{raw_data_outpath}/feature_correlation.png')
            data_plots.graph_feature_cdf(X_df, 
                                             f'{raw_data_outpath}/feature_cdf.png')
    else:  # Multi-label
        if not os.path.exists(f'{raw_data_outpath}/feature_heatmap.png'):
            # Handle multi-label plotting differently if necessary
            # data_plots.graph_class_imbalance_multilabel(Y_df, 
            #                                            f'{raw_data_outpath}/class_imbalance.png')
            # data_plots.graph_feature_heatmap_multilabel(X_df, Y_df,
            #                                            f'{raw_data_outpath}/feature_heatmap.png')
            pass



def inspect_pickle_content(pkl_path):
    """
    Inspect contents of a pickle file, showing structure and samples
    """
    print(f"\nInspecting pickle file: {pkl_path}")
    print("=" * 80)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    print("\n1. Basic Info:")
    print(f"Type of loaded data: {type(data)}")
    print(f"Is empty? {not bool(data)}")
    
    if isinstance(data, dict):
        print(f"\n2. Dictionary Structure:")
        print(f"Number of top-level keys: {len(data)}")
        print("\nTop-level keys:")
        for key in list(data.keys())[:5]:  # First 5 keys
            print(f"- {key} ({type(key)})")
            
        # Sample a random key for deeper inspection
        if data:
            sample_key = next(iter(data))
            print(f"\n3. Sample value for key '{sample_key}':")
            sample_value = data[sample_key]
            print(f"Type: {type(sample_value)}")
            
            # If the value is also a dictionary, show its structure
            if isinstance(sample_value, dict):
                print("Nested dictionary structure:")
                for k, v in list(sample_value.items())[:3]:  # First 3 items
                    print(f"- {k}: {type(v)}")
                    if isinstance(v, (dict, list)):
                        print(f"  Length: {len(v)}")
                    try:
                        print(f"  Sample: {str(v)[:100]}...")  # First 100 chars
                    except:
                        print("  Sample: [Cannot display sample]")
            
            # If it's a list or array, show some info
            elif isinstance(sample_value, (list, np.ndarray)):
                print(f"Length: {len(sample_value)}")
                print("First few elements:")
                print(sample_value[:3])
    
    # If it's not a dictionary, show appropriate info
    else:
        if isinstance(data, (list, np.ndarray)):
            print(f"\nArray/List Info:")
            print(f"Length: {len(data)}")
            print("First few elements:")
            print(data[:3])
        else:
            print("\nData sample:")
            try:
                print(str(data)[:200])  # First 200 chars
            except:
                print("[Cannot display data sample]")