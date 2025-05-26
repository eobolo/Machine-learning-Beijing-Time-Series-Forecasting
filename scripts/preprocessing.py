import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def explore_data(train, test, output_dir='outputs'):
    """
    Perform exploratory data analysis: summary statistics and visualizations.
    Save plots to the output directory.
    """
    print("Training Data Shape:", train.shape)
    print("Test Data Shape:", test.shape)
    print("\nTraining Data Columns:", train.columns.tolist())
    print("\nMissing Values in Training Data:\n", train.isnull().sum())
    print("\nMissing Values in Test Data:\n", test.isnull().sum())
    print("\nTraining Data Summary Statistics:\n", train.describe())

    # Plot PM2.5 time series with NaN locations
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['pm2.5'], label='PM2.5 Original')
    nan_indices = train[train['pm2.5'].isnull()].index
    plt.scatter(nan_indices, [0]*len(nan_indices), color='red', label='NaN Locations', alpha=0.6)
    plt.title('PM2.5 Levels Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.savefig(f'{output_dir}/pm25_time_series.png')
    plt.close()

    # Plot smoothed PM2.5 trend (120-hour moving average)
    window_size = 120
    pm25_smoothed = train['pm2.5'].rolling(window=window_size).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(pm25_smoothed.index, pm25_smoothed, label=f'PM2.5 Smoothed (Window={window_size})', color='red')
    plt.title('PM2.5 Smoothed Levels Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('PM2.5 (Smoothed)')
    plt.legend()
    plt.savefig(f'{output_dir}/pm25_smoothed.png')
    plt.close()

    # Correlation heatmap (excluding 'No')
    columns_for_correlation = train.drop(['No'], axis=1)
    correlation_matrix = columns_for_correlation.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Features')
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()

def handle_missing_data(train, test):
    """
    Handle missing values using cubic spline interpolation and backfill.
    """
    train.interpolate(method='cubicspline', inplace=True)
    test.interpolate(method='cubicspline', inplace=True)
    train.bfill(inplace=True)
    test.bfill(inplace=True)
    print("\nMissing Values After Imputation (Training):\n", train.isnull().sum())
    print("\nMissing Values After Imputation (Test):\n", test.isnull().sum())

def create_sequences(X_train, y_train, X_test, seq_length=12, val_split=0.2, peer_inspired=False):
    """
    Preprocess data by scaling features and targets, creating sequences.
    Supports peer-inspired sequence creation for test data if specified.
    """
    # Extract datetime index
    train_datetime = X_train.index
    test_datetime = X_test.index

    # Convert to NumPy arrays
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()

    # Align lengths of X_train and y_train
    if len(X_train_np) != len(y_train_np):
        print(f"Warning: Length mismatch between X_train ({len(X_train_np)}) and y_train ({len(y_train_np)}).")
        min_len = min(len(X_train_np), len(y_train_np))
        X_train_np = X_train_np[:min_len]
        y_train_np = y_train_np[:min_len]
        train_datetime = train_datetime[:min_len]
    else:
        print(f"X_train and y_train lengths are consistent: {len(X_train_np)} samples.")

    # Initialize scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Scale input features
    X_train_scaled = scaler_X.fit_transform(X_train_np)
    X_test_scaled = scaler_X.transform(X_test_np)

    # Scale target values
    y_train_scaled = scaler_y.fit_transform(y_train_np.reshape(-1, 1))

    # Create training sequences
    n_features = X_train_scaled.shape[1]
    train_sequences = []
    train_targets = []
    for i in range(len(X_train_scaled) - seq_length + 1):
        seq = X_train_scaled[i:i + seq_length]
        target = y_train_scaled[i + seq_length - 1]
        train_sequences.append(seq)
        train_targets.append(target)
    train_sequences = np.array(train_sequences)
    train_targets = np.array(train_targets)

    # Create test sequences
    if peer_inspired:
        # Peer-inspired method for test sequences (Models 6 and 7)
        X_combined_scaled = np.vstack((X_train_scaled[-(seq_length-1):], X_test_scaled))
        n_test = len(X_test_scaled)
        X_test_seq = []
        for i in range(n_test):
            start_idx = i
            end_idx = i + seq_length
            if end_idx <= len(X_combined_scaled):
                X_test_seq.append(X_combined_scaled[start_idx:end_idx])
        X_test_seq = np.array(X_test_seq)
    else:
        # Standard method for test sequences (Models 1–5)
        overlap_rows = X_train_scaled[-(seq_length-1):]
        extended_test_scaled = np.vstack((overlap_rows, X_test_scaled))
        test_sequences = []
        test_datetimes = []
        for i in range(len(extended_test_scaled) - seq_length + 1):
            seq = extended_test_scaled[i:i + seq_length]
            test_sequences.append(seq)
            test_datetimes.append(test_datetime[i])
        X_test_seq = np.array(test_sequences)
        test_datetimes = test_datetime.tolist()
        X_test_seq = X_test_seq[-len(X_test_scaled):]
        test_datetime = test_datetimes[-len(X_test_scaled):]

    # Validate test sequence length
    if len(X_test_seq) != len(X_test_scaled):
        raise ValueError(f"Mismatch between test sequences ({len(X_test_seq)}) and X_test ({len(X_test_scaled)}).")
    print(f"Number of test sequences: {len(X_test_seq)}")
    print(f"Number of test datetimes: {len(test_datetime)}")

    # Split training sequences into train and validation
    if len(train_sequences) == 0 or len(train_targets) == 0:
        raise ValueError("Not enough training sequences.")
    if len(train_sequences) < 2:
        raise ValueError("Insufficient training sequences for validation split.")
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
        train_sequences, train_targets, test_size=val_split, random_state=42, shuffle=False
    )

    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, test_datetime, scaler_X, scaler_y