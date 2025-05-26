import os
from data_loading import load_data, initial_processing
from preprocessing import explore_data, handle_missing_data, create_sequences
from models import build_lstm_model, build_bilstm_model
from training import train_model, predict_and_submit

def run_model(model_number, X_train, y_train, X_test, output_dir='outputs'):
    """
    Run the pipeline for a specific model.
    """
    print(f"\nRunning Model {model_number}...")
    
    # Define model-specific parameters
    if model_number == 1:
        seq_length = 12
        units = 25
        learning_rate = 0.0001
        activation = 'sigmoid'
        batch_norm = False
        peer_inspired = False
        model_name = 'model1_seq12'
        model_func = build_lstm_model
    elif model_number == 2:
        seq_length = 24
        units = 25
        learning_rate = 0.0001
        activation = 'sigmoid'
        batch_norm = False
        peer_inspired = False
        model_name = 'model2_seq24'
        model_func = build_lstm_model
    elif model_number == 3:
        seq_length = 48
        units = 50
        learning_rate = 0.0001
        activation = 'sigmoid'
        batch_norm = False
        peer_inspired = False
        model_name = 'model3_seq48'
        model_func = build_lstm_model
    elif model_number == 4:
        seq_length = 48
        units = 50
        learning_rate = 0.00001
        activation = 'sigmoid'
        batch_norm = True
        peer_inspired = False
        model_name = 'model4_seq48_bn'
        model_func = build_lstm_model
    elif model_number == 5:
        seq_length = 48
        units = 50
        learning_rate = 0.00001
        activation = 'sigmoid'
        batch_norm = True
        peer_inspired = False
        model_name = 'model5_bilstm'
        model_func = build_bilstm_model
    elif model_number == 6:
        seq_length = 48
        units = 50
        learning_rate = 0.00001
        activation = 'sigmoid'
        batch_norm = True
        peer_inspired = True
        model_name = 'model6_bilstm_peer'
        model_func = build_bilstm_model
    elif model_number == 7:
        seq_length = 48
        units = 50
        learning_rate = 0.00001
        activation = 'linear'
        batch_norm = True
        peer_inspired = True
        model_name = 'model7_bilstm_linear'
        model_func = build_bilstm_model
    else:
        raise ValueError(f"Model {model_number} not defined.")

    # Preprocess data
    X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, test_datetimes, scaler_X, scaler_y = create_sequences(
        X_train, y_train, X_test, seq_length=seq_length, val_split=0.2, peer_inspired=peer_inspired
    )

    # Build model
    model = model_func(
        n_features=X_train_seq.shape[2],
        seq_length=seq_length,
        units=units,
        learning_rate=learning_rate,
        activation=activation,
        batch_norm=batch_norm
    )

    # Train model
    train_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, output_dir, model_name)

    # Predict and create submission
    predict_and_submit(model, X_test_seq, test_datetimes, scaler_y, output_dir, model_name)

def main():
    """
    Main function to run the entire pipeline for all models.
    """
    # Create output directory if it doesn't exist
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    train, test, _ = load_data()

    # Initial processing
    X_train, y_train, X_test = initial_processing(train, test)

    # Explore data and handle missing values
    explore_data(train, test, output_dir)
    handle_missing_data(train, test)

    # Run each model
    for model_number in range(1, 8):
        run_model(model_number, X_train, y_train, X_test, output_dir)

if __name__ == "__main__":
    main()