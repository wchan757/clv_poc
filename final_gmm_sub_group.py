import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class CustomerSegmentation:
    def __init__(self, file_path, features=None):
        """Initialize with file path and features for clustering."""
        self.file_path = file_path
        self.features = features if features else [
            'USAGE_DATA_MB_ROLLING_120DAYS_QUANTITY',
            'USAGE_VOICE_INTRA_CONSUMABLE_MINS_90DAYS',
            'CLV_OVERALL_REVENUE',
            'TENURE_COUNT_MOS',
            'O_pred', 'C_pred', 'E_pred', 'A_pred', 'N_pred'
        ]
        self.df = None
        self.existing_features = None

    def load_data(self):
        """Load the dataset."""
        self.df = pd.read_csv(self.file_path)
        print("CSV file loaded successfully.")
        
        # Check which features exist in the DataFrame
        self.existing_features = [col for col in self.features if col in self.df.columns]
        if len(self.existing_features) != len(self.features):
            missing_features = set(self.features) - set(self.existing_features)
            print(f"Warning: Some features are missing from the dataset: {missing_features}")
        
        return self

    def preprocess_data(self):
        """Preprocess the data by handling missing values and scaling."""
        print("Checking for missing values before imputation...")
        missing_counts = self.df[self.existing_features].isna().sum()
        print("Missing values per feature:\n", missing_counts)
        
        features_data = self.df[self.existing_features].copy()
        all_missing_features = features_data.columns[features_data.isna().all()].tolist()
        features_to_impute = [f for f in self.existing_features if f not in all_missing_features]
        
        if all_missing_features:
            print(f"Warning: Features with all missing values: {all_missing_features}")
            for feature in all_missing_features:
                self.df[feature] = 0
        
        if features_to_impute:
            imputer = SimpleImputer(strategy="mean")
            imputed_data = imputer.fit_transform(features_data[features_to_impute])
            imputed_df = pd.DataFrame(imputed_data, columns=features_to_impute, index=self.df.index)
            self.df[features_to_impute] = imputed_df
        
        if self.df[self.existing_features].isna().sum().sum() > 0:
            print("Warning: Some NaN values still exist after imputation!")
        
        # Apply feature scaling
        scaler = StandardScaler()
        self.df[self.existing_features] = scaler.fit_transform(self.df[self.existing_features])
        print("Data normalization complete.")
        
        return self

    def apply_gmm(self, n_subgroups=3):
        """Apply Gaussian Mixture Model clustering per TUS_Group_Name."""
        df_results = []
        
        for group_name, group_df in self.df.groupby("TUS_Group_Name"):
            if len(group_df) >= n_subgroups:
                print(f"Running GMM for {group_name}...")
                gmm = GaussianMixture(n_components=n_subgroups, covariance_type='tied', random_state=42)
                labels = gmm.fit_predict(group_df[self.existing_features])
                group_df = group_df.copy()
                group_df["GMM_TUS_Subgroup"] = labels + 1  # Shift values to range 1-3
            else:
                print(f"Skipping GMM for {group_name} (insufficient data)")
                group_df = group_df.copy()
                group_df["GMM_TUS_Subgroup"] = -1
            
            df_results.append(group_df)
        
        self.df = pd.concat(df_results)
        return self

    def save_results(self, output_path):
        """Save the processed dataset."""
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        return self

    def process(self, output_path):
        """Run the complete processing pipeline."""
        self.load_data()
        self.preprocess_data()
        
        # Verify no NaNs before clustering
        if self.df[self.existing_features].isna().sum().sum() > 0:
            raise ValueError("ERROR: NaN values still present after imputation!")
        
        self.apply_gmm()
        self.save_results(output_path)
        return self

    def get_data(self):
        """Return the processed DataFrame."""
        return self.df

def main():
    input_file = r"C:\Users\13032\Desktop\ph_behavior\big_5_personality\10k_predictions_with_tus_negative_possible_with_group.csv"
    output_file = "output_final.csv"
    
    # Create and run the segmentation
    segmentation = CustomerSegmentation(input_file)
    segmentation.process(output_file)

if __name__ == "__main__":
    main()
