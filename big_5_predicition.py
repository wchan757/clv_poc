import os
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

class MyOptunaLightGBMEasy:
    def __init__(
        self,
        csv_path="C:\\Users\\13032\\Desktop\\ph_behavior\\combined_data_train.csv",
        target_columns=["O", "C", "E", "A", "N"],
        random_seed=42
    ):
        self.random_seed = random_seed
        self.target_columns = target_columns
        self.models_folder = "models"
        os.makedirs(self.models_folder, exist_ok=True)

        print(f"Loading data from: {csv_path}")
        self.df = pd.read_csv(csv_path)

        # Drop SUBSCRIBER_ID if it exists
        if 'SUBSCRIBER_ID' in self.df.columns:
            self.df.drop(columns=['SUBSCRIBER_ID'], inplace=True)
            print("'SUBSCRIBER_ID' column removed.")

        # 1. Convert target columns to numeric (invalid entries become NaN)
        for target in self.target_columns:
            self.df[target] = pd.to_numeric(self.df[target], errors='coerce')

        # 2. Drop rows with NaN in any target column
        before_dropna = len(self.df)
        self.df.dropna(subset=self.target_columns, inplace=True)
        after_dropna = len(self.df)
        print(f"Dropped {before_dropna - after_dropna} rows where any target was non-numeric or missing.")

        # 3. Keep only rows where each target is in [1..5]
        for target in self.target_columns:
            valid_mask = self.df[target].between(1, 5, inclusive='both')
            initial_count = len(self.df)
            self.df = self.df[valid_mask]
            final_count = len(self.df)
            removed = initial_count - final_count
            if removed > 0:
                print(f"Removed {removed} rows from target '{target}' outside 1..5.")

        # 4. Shift labels from [1..5] to [0..4] for LightGBM multiclass
        for target in self.target_columns:
            self.df[target] = self.df[target].astype(int)
            self.df[target] = self.df[target] - 1  # Now 0..4

        # 5. Convert any remaining object columns (not target) into numeric codes
        object_cols = [
            col for col in self.df.columns
            if (self.df[col].dtype == "object") and (col not in self.target_columns)
        ]
        for col in object_cols:
            print(f"Converting column '{col}' from object to numeric codes.")
            self.df[col] = self.df[col].astype('category').cat.codes

        # Ensure target columns are int
        for target in self.target_columns:
            self.df[target] = self.df[target].astype(int)

        print(f"Data types after cleaning:\n{self.df.dtypes}")

    def run_study_and_train(self, n_trials=10):
        for target in self.target_columns:
            print(f"\nTraining model for target: {target}")

            # Separate features (X) and label (y)
            X = self.df.drop(columns=self.target_columns)
            y = self.df[target]

            # Check if stratification is possible
            if y.nunique() < 2 or y.value_counts().min() < 2:
                print("Warning: Not enough samples per class for stratification. Using random split.")
                stratify = None
            else:
                stratify = y

            # ===== Outer Split: Reserve 20% as final hold-out test set =====
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_seed, stratify=stratify
            )

            # ========= K-Fold CV inside the objective for hyperparameter tuning ========
            def objective(trial):
                params = {
                    'objective': 'multiclass',
                    'num_class': len(y.unique()),  # Should be 5 if labels are now 0..4
                    'metric': 'multi_logloss',
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'verbose': -1,
                    'seed': self.random_seed
                }

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
                fold_aucs = []
                fold_iterations = []

                for train_idx, valid_idx in skf.split(X_train, y_train):
                    X_tr = X_train.iloc[train_idx]
                    y_tr = y_train.iloc[train_idx]
                    X_val = X_train.iloc[valid_idx]
                    y_val = y_train.iloc[valid_idx]

                    dtrain = lgb.Dataset(X_tr, label=y_tr)
                    dvalid = lgb.Dataset(X_val, label=y_val)

                    model = lgb.train(
                        params,
                        dtrain,
                        valid_sets=[dvalid],
                        num_boost_round=1000,
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=50),
                            lgb.log_evaluation(period=0)  # quiet logging
                        ]
                    )

                    preds_val = model.predict(X_val, num_iteration=model.best_iteration)
                    auc_val = roc_auc_score(
                        pd.get_dummies(y_val),
                        preds_val,
                        multi_class='ovr',
                        average='macro'
                    )
                    fold_aucs.append(auc_val)
                    fold_iterations.append(model.best_iteration)

                # Average AUC across folds
                mean_auc = np.mean(fold_aucs)

                # Save average best_iteration
                avg_best_iter = int(np.mean(fold_iterations))
                trial.set_user_attr('n_estimators', avg_best_iter)

                return mean_auc

            # =============== Run Optuna Study (maximize AUC) ==================
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            best_params = study.best_trial.params
            best_params.update({
                'objective': 'multiclass',
                'num_class': len(y.unique()),
                'metric': 'multi_logloss',
                'verbose': -1,
                'seed': self.random_seed
            })

            # ======= Final Retraining on the entire training set (80%) ========
            best_n_estimators = study.best_trial.user_attrs.get('n_estimators', 100)
            best_model = lgb.train(
                best_params,
                lgb.Dataset(X_train, label=y_train),
                num_boost_round=best_n_estimators
            )

            # ======= Evaluate on the hold-out test set ========
            preds_test = best_model.predict(X_test, num_iteration=best_n_estimators)
            test_auc = roc_auc_score(
                pd.get_dummies(y_test),
                preds_test,
                multi_class='ovr',
                average='macro'
            )

            print(f"Final Test AUC for target '{target}': {test_auc:.4f}")

            # ====== Save the final model for this target ======
            model_filename = os.path.join(self.models_folder, f"{target}_best_model.pkl")
            with open(model_filename, "wb") as f:
                pickle.dump(best_model, f)

            print(f"Model for {target} saved to {model_filename}\n")

def main():
    experiment = MyOptunaLightGBMEasy()
    experiment.run_study_and_train(n_trials=10)

if __name__ == "__main__":
    main()
