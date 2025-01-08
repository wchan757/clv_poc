# README

## Overview

This Python script implements a machine learning pipeline to train and evaluate models on a dataset. The script uses three models:
- **Linear Regression** (Baseline Model)
- **Random Forest Regressor** (with hyperparameter tuning using Optuna)
- **LightGBM Regressor** (with hyperparameter tuning using Optuna)

The script also includes the following functionality:
1. Data preprocessing, including handling categorical features using one-hot encoding.
2. Hyperparameter optimization for Random Forest and LightGBM using **Optuna**.
3. Logging of key events and results to both the console and a log file (`model_training.log`).
4. Saving trained models to disk for future use.

---

## Features

- **Data Preprocessing**:
  - Handles categorical and numerical features.
  - Automatically aligns feature columns between training and testing datasets.
  - Drops unnecessary columns if specified.

- **Model Training**:
  - Baseline Linear Regression.
  - Random Forest with Optuna hyperparameter tuning.
  - LightGBM with Optuna hyperparameter tuning and early stopping.

- **Logging**:
  - Logs are saved to `model_training.log` for detailed tracking.

- **Model Saving**:
  - Trained models are saved in the `models/` directory.

---

## Requirements

To run the script, the following dependencies are required:

### Python Version
- **Python >= 3.8**

### Required Libraries
Install the dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:
```plaintext
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
lightgbm==3.3.5
optuna==3.3.0
```

---

## File Structure

- **`main.py`**: The script containing the pipeline logic.
- **`requirements.txt`**: The file containing all required dependencies.
- **`models/`**: The directory where trained models are saved.
- **`model_training.log`**: The log file capturing detailed execution logs.

---

## How to Run

1. **Prepare Your Dataset**:
   - Create two CSV files:
     - `2023.csv` (Training data)
     - `2024.csv` (Testing data)
   - Each dataset should include:
     - A target variable column (e.g., `lifetime_value`).
     - Numerical and categorical features.

2. **Update the Script**:
   - Modify the `main` function to specify:
     - Paths to the `train_file` and `test_file`.
     - The name of the target column.
     - Any categorical columns or columns to be dropped.

3. **Run the Script**:
   Execute the script as follows:

   ```bash
   python main.py
   ```

4. **Results**:
   - The script will preprocess the data, train the models, and save the trained models in the `models/` directory.
   - Training and evaluation metrics (MSE for each model) will be logged to `model_training.log`.

---

## Expected Dataset Format

Both `2023.csv` (training data) and `2024.csv` (testing data) should have the following structure:

| customer_id | feature1 | feature2 | category1 | category2 | lifetime_value |
|-------------|----------|----------|-----------|-----------|----------------|
| CUST_1      | 0.5      | 50.3     | A         | X         | 100.0          |
| CUST_2      | 1.2      | 40.2     | B         | Y         | 200.5          |
| CUST_3      | 0.7      | 45.1     | C         | Z         | 150.3          |

- **`lifetime_value`** is the target variable.
- **`customer_id`** is an identifier and can be excluded from training by specifying it in the `drop_columns` parameter.

---

## Script Workflow

1. **Data Preprocessing**:
   - Reads the CSV files for training and testing data.
   - Handles missing columns and encodes categorical variables using one-hot encoding.

2. **Training Models**:
   - **Linear Regression**: Trains a baseline model with no hyperparameter tuning.
   - **Random Forest**: Tunes hyperparameters using Optuna and trains the best model.
   - **LightGBM**: Tunes hyperparameters using Optuna and trains the best model with early stopping.

3. **Evaluation**:
   - Calculates the Mean Squared Error (MSE) for each model on the testing dataset.
   - Logs the results to `model_training.log`.

4. **Saving Models**:
   - Saves the trained models to the `models/` directory as `.pkl` files.

---

## Example Output

Upon successful execution, the following results will be logged in `model_training.log`:

```plaintext
2025-01-08 10:00:00 [INFO] Loading datasets...
2025-01-08 10:00:01 [INFO] Training dataset shape: (1000, 12), Target shape: (1000,)
2025-01-08 10:00:01 [INFO] Testing dataset shape: (200, 12), Target shape: (200,)
2025-01-08 10:00:01 [INFO] Datasets loaded and preprocessed successfully.
2025-01-08 10:00:02 [INFO] Training baseline Linear Regression model...
2025-01-08 10:00:03 [INFO] Baseline Linear Regression MSE: 150.8721
2025-01-08 10:00:04 [INFO] Starting Random Forest training with Optuna hyperparameter tuning...
2025-01-08 10:00:30 [INFO] Best Random Forest parameters: {'n_estimators': 500, 'max_depth': 10, ...}
2025-01-08 10:00:30 [INFO] Best Random Forest MSE: 120.5678
2025-01-08 10:01:00 [INFO] Best LightGBM MSE: 110.4567
2025-01-08 10:01:00 [INFO] Model training complete.
```

Trained models will be saved as:
- `models/linear_regression.pkl`
- `models/random_forest.pkl`
- `models/lightgbm.pkl`

---

## Customization

### **Adjusting Hyperparameters**
- **Random Forest**: Modify the search space in the `rf_objective` function.
- **LightGBM**: Modify the search space in the `lgb_objective` function.

### **Changing Early Stopping**:
- Adjust the `EARLY_STOPPING_ROUNDS` constant to control early stopping for LightGBM.

---

## Troubleshooting

1. **Missing Dependencies**:
   - Ensure all required libraries are installed using `pip install -r requirements.txt`.

2. **Mismatch in Categorical Columns**:
   - Ensure categorical columns exist in both the training and testing datasets.

3. **FileNotFoundError**:
   - Verify the paths to the `2023.csv` and `2024.csv` files.

---

## Future Enhancements

- Add support for cross-validation.
- Implement additional models for comparison.
- Introduce feature importance analysis.

---

Feel free to reach out if you need further assistance!
