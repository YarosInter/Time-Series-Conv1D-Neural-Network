# Time-Series Conv1D Neural Network

This repository focuses on time series analysis using a 1D Convolutional Neural Network (Conv1D) applied to financial data. The project aims to forecast future price movements using deep learning techniques and evaluate the model's performance on financial datasets.

## Project Overview

Time series analysis is essential for predicting financial data, particularly for trading and investment strategies. This project leverages historical financial data, applies preprocessing techniques, and uses a Conv1D neural network to predict price movements. Conv1D models are effective for detecting local patterns and trends in time series data.

### Key Features

- **Data Retrieval**: Custom functions to download historical financial data.
- **Data Processing**: Includes creating lagged features and transforming data into the appropriate format for Conv1D models.
- **Conv1D Model Development**: The Conv1D neural network is implemented using TensorFlow and Keras, consisting of:
  - Input layer for 1D time series data
  - Multiple Conv1D layers with ReLU activation and Dropout layers for regularization
  - Global Average Pooling for dimensionality reduction
  - Output layer to predict price direction
  - **EarlyStopping**: Monitors validation loss and halts training to prevent overfitting
  - **ModelCheckpoint**: Saves the best-performing model based on validation metrics.
- **Backtesting and Performance Evaluation**: Financial metrics such as Sortino, Beta, and Alpha ratios are calculated to assess model performance. Cumulative returns and drawdowns are visualized.
- **Visualization**: Functions are provided to plot cumulative returns, test predictions, and drawdowns.

## Repository Structure

- **`data.py`**: Contains functions for retrieving and processing financial data:
  - `get_rates`: Retrieves historical price data for a given asset and timeframe.
  - `add_shifted_columns`: Adds lagged features to the dataset for time series prediction.
  - `split_data`: Splits the dataset into training, validation, and test sets.

- **`backtest.py`**: Provides functions for backtesting financial strategies:
  - `compute_strategy_returns`: Computes cumulative returns for a given strategy.
  - `compute_drawdown`: Calculates drawdown for a strategy.
  - `vectorize_backtest_returns`: Computes financial ratios such as Sortino, Beta, and Alpha.

- **`ai.py`**: Contains the Conv1D model training function:
  - `run_conv1d`: Trains the Conv1D model using the processed data, including callbacks for model saving and early stopping.

- **`display.py`**: Contains functions to visualize results:
  - `plot_test_returns`: Plots cumulative returns on the test dataset.
  - `plot_drawdown`: Visualizes the model's drawdown over time.

- **`Conv1D_Layer_Neural_Network.ipynb`**: Jupyter Notebook demonstrating the steps for building, training, and evaluating the Conv1D Neural Network. The notebook includes:
  - Data processing
  - Model creation and compilation with TensorFlow and Keras
  - Training the model with EarlyStopping and ModelCheckpoint
  - Evaluating the model on the validation set
  - Backtesting on financial data

- **`.gitignore`**: Lists files and directories to be ignored by Git.

- **`README.md`**: This file.


## Installation

To run this project, you need Python installed along with the following dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tensorflow`


## How to View

**Clone the Repository for Viewing**: You may clone this repository to your local machine for personal review and educational purposes only:
   ```bash 
   git clone https://github.com/YarosInter/Time-Series-Conv1D-Neural-Network.git
   ```
   
   
## Contributing

This repository is for personal or educational purposes only. Contributions and modifications are permitted with explicitly allowed. Feel free to reach out if you'd like to collaborate or contribute, let’s build a trustable and functional model together!


## Disclaimer

The code in this repository is for educational or personal review only and is not licensed for use, modification, or distribution. 
This code is part of my journey in learning and experimenting with new ideas. It’s shared for educational purposes and personal review. Please feel free to explore, but kindly reach out for permission if you’d like to use, modify, or distribute any part of it.
