import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath='sdn_traffic_data.csv', sequence_length=10):
    """
    Loads data, performs feature engineering, scaling, and sequencing for LSTM.
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    
    # 1. Feature Engineering
    # Rolling averages to capture trends
    df['packet_in_rolling_mean'] = df['packet_in_count'].rolling(window=5, min_periods=1).mean()
    df['cpu_rolling_mean'] = df['cpu_utilization'].rolling(window=5, min_periods=1).mean()
    
    # Lag features (past values)
    df['packet_in_lag1'] = df['packet_in_count'].shift(1).fillna(0)
    df['cpu_lag1'] = df['cpu_utilization'].shift(1).fillna(0)

    # 2. Select Features for Model
    feature_cols = [
        'packet_in_count', 'flow_table_usage', 'active_switches', 
        'cpu_utilization', 'memory_utilization', 'protocol_type',
        'packet_in_rolling_mean', 'cpu_rolling_mean',
        'packet_in_lag1', 'cpu_lag1'
    ]
    target_col = 'load_label' # Classification target
    # or use 'cpu_utilization' as regression target if predicting load value

    # 3. Scaling
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print("Feature scaling complete.")
    
    # 4. Create Sequences for LSTM/GRU (if using temporal models)
    # X shape: (samples, sequence_length, features)
    # y shape: (samples, )
    X, y = [], []
    data_matrix = df[feature_cols].values
    target_values = df[target_col].values

    for i in range(len(df) - sequence_length):
        X.append(data_matrix[i : i + sequence_length])
        y.append(target_values[i + sequence_length])
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Data sequences created. Shape: {X.shape}")
    
    return X, y, scaler

if __name__ == "__main__":
    load_and_preprocess_data()
