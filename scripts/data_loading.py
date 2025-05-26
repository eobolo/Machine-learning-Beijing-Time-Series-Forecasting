import os
import pandas as pd
from google.colab import drive

def load_data():
    """
    Load train and test datasets from Google Drive or local Colab environment.
    Returns the train and test DataFrames, along with a flag indicating the source.
    """
    # Define file paths for Google Drive and local Colab
    drive_train_path = '/content/drive/MyDrive/Kaggle_competition_ML/air_quality_forcasting/train.csv'
    drive_test_path = '/content/drive/MyDrive/Kaggle_competition_ML/air_quality_forcasting/test.csv'
    local_train_path = '/content/train.csv'
    local_test_path = '/content/test.csv'

    train = None
    test = None
    loaded_from_drive = False

    # Try loading from Google Drive first
    try:
        print("Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        print("Attempting to load data from Google Drive...")
        if os.path.exists(drive_train_path) and os.path.exists(drive_test_path):
            train = pd.read_csv(drive_train_path)
            test = pd.read_csv(drive_test_path)
            loaded_from_drive = True
            print("Successfully loaded data from Google Drive.")
        else:
            print("Files not found in Google Drive path. Trying local Colab environment...")
    except Exception as e:
        print(f"Error loading from Google Drive: {e}. Trying local Colab environment...")

    # If not loaded from Drive, try local Colab environment
    if train is None or test is None:
        try:
            print("Attempting to load data from local Colab environment...")
            if os.path.exists(local_train_path) and os.path.exists(local_test_path):
                train = pd.read_csv(local_train_path)
                test = pd.read_csv(local_test_path)
                print("Successfully loaded data from local Colab environment.")
            else:
                print("Files not found in local Colab environment.")
        except Exception as e:
            print(f"Error loading from local Colab environment: {e}")

    # Check if data was loaded successfully
    if train is None or test is None:
        print("\nERROR: Data loading failed.")
        print("Please ensure you have uploaded 'train.csv' and 'test.csv' either:")
        print(f"1. To the following path in your Google Drive: {drive_train_path} and {drive_test_path}")
        print("   (and ensure Google Drive is mounted)")
        print("OR")
        print("2. Directly upload 'train.csv' and 'test.csv' to the Colab environment.")
        raise FileNotFoundError("Data files not found.")
    else:
        if loaded_from_drive:
            print("\nData loaded successfully from Google Drive.")
        else:
            print("\nData loaded successfully from local Colab environment.")

    return train, test, loaded_from_drive

def initial_processing(train, test):
    """
    Perform initial processing: convert datetime column and set as index.
    Drop 'No' column and separate features and target.
    """
    # Convert 'datetime' column to datetime format
    train['datetime'] = pd.to_datetime(train['datetime'])
    test['datetime'] = pd.to_datetime(test['datetime'])

    # Set 'datetime' as index
    train.set_index('datetime', inplace=True)
    test.set_index('datetime', inplace=True)

    # Separate features and target, drop 'No' column
    X_train = train.drop(['pm2.5', 'No'], axis=1)
    y_train = train['pm2.5']
    X_test = test.drop(['No'], axis=1)

    return X_train, y_train, X_test