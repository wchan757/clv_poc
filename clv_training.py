import os
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna

# Set up logging
logging.basicConfig(
    filename="model_training.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def setup_logger():
    """Set up a logger to print messages to the console and save to a log file."""
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

setup_logger()

# Define constants
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 42
MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Function to load data
def load_and_preprocess_data(train_file, test_file, target_column, categorical_columns=None, drop_columns=None):
    """
    Load real data for 2023 (training) and 2024 (testing), preprocess it with proper handling 
    of categorical and numerical variables, and split into features (X) and target (y).

    Args:
        train_file (str): Path to the 2023 training dataset file.
        test_file (str): Path to the 2024 testing dataset file.
        target_column (str): Name of the target variable column.
        categorical_columns (list of str, optional): List of categorical column names to encode.
        drop_columns (list of str, optional): List of columns to drop from the dataset.

    Returns:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Target for training.
        X_test (pd.DataFrame): Features for testing.
        y_test (pd.Series): Target for testing.
    """
    logging.info("Loading datasets...")

    # Load the training and testing data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Validate that the target column exists in both datasets
    if target_column not in train_data.columns or target_column not in test_data.columns:
        raise ValueError(f"Target column '{target_column}' must be present in both datasets.")

    # Drop unnecessary columns if specified
    if drop_columns:
        train_data.drop(columns=drop_columns, inplace=True, errors="ignore")
        test_data.drop(columns=drop_columns, inplace=True, errors="ignore")

    # Separate features (X) and target (y)
    y_train = train_data[target_column]
    y_test = test_data[target_column]
    X_train = train_data.drop(columns=[target_column])
    X_test = test_data.drop(columns=[target_column])

    # Handle categorical variables
    if categorical_columns:
        logging.info("Encoding categorical variables...")
        # Ensure all specified categorical columns exist in both datasets
        for col in categorical_columns:
            if col not in X_train.columns or col not in X_test.columns:
                raise ValueError(f"Categorical column '{col}' must be present in both datasets.")

        # Use one-hot encoding for categorical variables
        X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

        # Align columns between train and test datasets, filling missing columns with 0
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Log preprocessing summary
    logging.info(f"Training dataset shape: {X_train.shape}, Target shape: {y_train.shape}")
    logging.info(f"Testing dataset shape: {X_test.shape}, Target shape: {y_test.shape}")
    logging.info("Datasets loaded and preprocessed successfully.")

    return X_train, y_train, X_test, y_test

# Function to train and evaluate a baseline model
def train_baseline(X_train, y_train, X_test, y_test):
    """
    Train a baseline Linear Regression model.
    """
    logging.info("Training baseline Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    logging.info(f"Baseline Linear Regression MSE: {mse:.4f}")
    save_model(model, "linear_regression.pkl")
    return mse

# Function to save a model to disk
def save_model(model, filename):
    path = os.path.join(MODEL_SAVE_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {path}")

# Optuna objective function for Random Forest
def rf_objective(trial, X_train, y_train, X_test, y_test):
    """
    Objective function for Optuna to tune Random Forest hyperparameters.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

    model = RandomForestRegressor(random_state=RANDOM_STATE, **params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Function to train and evaluate Random Forest with Optuna
def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model with Optuna hyperparameter tuning.
    """
    logging.info("Starting Random Forest training with Optuna hyperparameter tuning...")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: rf_objective(trial, X_train, y_train, X_test, y_test), n_trials=50)

    logging.info(f"Best Random Forest parameters: {study.best_params}")
    logging.info(f"Best Random Forest MSE: {study.best_value:.4f}")

    # Train final model with the best parameters
    best_params = study.best_params
    final_model = RandomForestRegressor(random_state=RANDOM_STATE, **best_params)
    final_model.fit(X_train, y_train)

    save_model(final_model, "random_forest.pkl")
    return study.best_value

# Optuna objective function for LightGBM
def lgb_objective(trial, X_train, y_train, X_test, y_test):
    """
    Objective function for Optuna to tune LightGBM hyperparameters.
    """
    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
        "n_estimators": 1000,  # Set a high value; early stopping will stop training earlier
    }

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    model = lgb.train(
        param,
        train_set,
        valid_sets=[train_set, valid_set],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Function to train and evaluate LightGBM
def train_lightgbm(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a LightGBM model with Optuna hyperparameter tuning.
    """
    logging.info("Starting LightGBM training with Optuna hyperparameter tuning...")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: lgb_objective(trial, X_train, y_train, X_test, y_test), n_trials=50)

    logging.info(f"Best LightGBM parameters: {study.best_params}")
    logging.info(f"Best LightGBM MSE: {study.best_value:.4f}")

    # Train final model with the best parameters
    best_params = study.best_params
    best_params["n_estimators"] = 1000
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    model = lgb.train(
        best_params,
        train_set,
        valid_sets=[train_set, valid_set],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=True,
    )

    save_model(model, "lightgbm.pkl")
    return study.best_value

# Main function
def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Train and evaluate models
    baseline_mse = train_baseline(X_train, y_train, X_test, y_test)
    rf_mse = train_random_forest(X_train, y_train, X_test, y_test)
    lgbm_mse = train_lightgbm(X_train, y_train, X_test, y_test)

    # Log final results
    logging.info("Model training complete.")
    logging.info(f"Baseline Linear Regression MSE: {baseline_mse:.4f}")
    logging.info(f"Random Forest MSE: {rf_mse:.4f}")
    logging.info(f"LightGBM MSE: {lgbm_mse:.4f}")

if __name__ == "__main__":
    main()
