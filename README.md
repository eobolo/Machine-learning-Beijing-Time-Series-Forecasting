## Project Overview

This repository contains the code and documentation for a time-series prediction project focused on forecasting PM2.5 levels (a key air quality metric) using LSTM-based models. The dataset spans from January 1, 2010, to December 31, 2014, with the training set covering January 1, 2010, to July 1, 2013, and the test set from July 2, 2013, to December 31, 2014. The goal was to predict hourly PM2.5 values, tackling challenges like temporal dependencies and distribution shifts between training and test sets. The project was developed as part of a Kaggle competition, experimenting with various LSTM architectures to minimize test RMSE.

The best-performing model (Model 6) achieved a test RMSE of 5001.7018 using a Bidirectional LSTM, though it highlighted areas for improvement, such as handling extreme values and further reducing overfitting.

## Repository Structure

The repository is organized into the following directories and files for clarity and reproducibility:

- data/: Folder for input data.
  - train.csv: Training data (January 1, 2010 - July 1, 2013).
  - test.csv: Test data (July 2, 2013 - December 31, 2014).
- outputs/: Stored model predictions, and submission files.
  - subm_bilstm.csv: Submission file for Model 5.
  - subm_model5_peer_test_sequence.csv: Submission file for Model 6.
  - subm_model5_linear_output.csv: Submission file for Model 7.
- report/: Contains project documentation and analysis.
  - Obolo Emmanuel Oluwapelumi PM2.5 Report.pdf: Full report detailing model design, experiment table, results, and implications.
- README.md: This file, providing an overview and instructions.
- requirements.txt: Lists Python dependencies for reproducibility.

## Data Sourcing

The dataset used in this project was sourced from a Kaggle competition focused on air quality forecasting, and the files are placed in the data/ directory as follows:
- data/train.csv
- data/test.csv

## Setup Instructions

To set up the environment and run the project, follow these steps:

1. Clone the Repository:
   - Run the following command in your terminal:
   ```bash
     git clone https://github.com/eobolo/Machine-learning-Beijing-Time-Series-Forecasting.git
   ```

2. Install Dependencies:
   - Ensure you have Python 3.8+ installed.
   - Install required packages using the provided requirements.txt:
   ```python
     pip install -r requirements.txt
    ```
   - The requirements.txt includes:
     numpy
     pandas
     matplotlib
     tensorflow
     scikit-learn

## Additional Notes

- The scripts assume the data files are in the data/ directory. Adjust file paths in the notebook if your setup differs.
- The project uses TensorFlow for LSTM implementation; ensure your environment supports it.