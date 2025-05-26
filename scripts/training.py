import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def train_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, output_dir='outputs', model_name='model'):
    """
    Train the model and plot training/validation metrics.
    """
    # Define callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50, batch_size=32, verbose=1,
        callbacks=[lr_scheduler, early_stopping]
    )

    # Plot training and validation metrics
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    if 'rmse_metric' in history.history:  # For Model 7 with RMSE metric
        plt.plot(history.history['rmse_metric'], label='Training RMSE (Scaled)')
        plt.plot(history.history['val_rmse_metric'], label='Validation RMSE (Scaled)')
    plt.title(f'Training and Validation Metrics ({model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/training_validation_loss_{model_name}.png')
    plt.close()

    return history

def predict_and_submit(model, X_test_seq, test_datetimes, scaler_y, output_dir='outputs', model_name='model'):
    """
    Make predictions on the test set and create a submission file.
    """
    # Predict on test set
    y_pred_scaled = model.predict(X_test_seq, verbose=1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    predictions = np.nan_to_num(y_pred)
    predictions = np.maximum(predictions, 0)
    predictions = np.round(predictions).astype(int)

    # Validate lengths
    if len(test_datetimes) != len(predictions):
        raise ValueError(f"Mismatch between test datetimes ({len(test_datetimes)}) and predictions ({len(predictions)}).")

    # Prepare submission file
    submission = pd.DataFrame({
        'row ID': pd.to_datetime(test_datetimes).strftime('%Y-%m-%d %-H:%M:%S'),
        'pm2.5': predictions.flatten()
    })

    # Debug: Print first few rows
    print(f"Submission DataFrame for {model_name} (first 10 rows):")
    print(submission.head(10))

    # Save submission file
    submission_path = f'{output_dir}/subm_{model_name}.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission CSV for {model_name} saved as '{submission_path}'")

    # Output sample predictions
    print(f"Sample predictions for {model_name}:")
    for i in range(min(5, len(predictions))):
        print(f"Sample {i+1}: Datetime = {submission['row ID'].iloc[i]}, Predicted PM2.5 = {predictions[i][0]}")

    return predictions