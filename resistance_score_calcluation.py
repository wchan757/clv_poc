import pandas as pd
from sklearn.preprocessing import StandardScaler

class ResistanceScoreCalculator:
    """
    A class to calculate the Resistance Score (RS_total) and Transaction Utility Score (TUS).
    
    Steps:
      1. Compute an 'internal_raw' resistance score based on personality traits.
      2. Standardize 'internal_raw' into 'internal_z'.
      3. Standardize external factors: 
         - TENURE_COUNT_MOS (tenure_z)
         - PAYMENT_CATEGORY_CODE (payment_raw -> payment_z)
      4. Compute the total resistance score RS_total (ensuring it's always > 0).
      5. Compute TUS = CLV_OVERALL_REVENUE / RS_total (allows negatives for CLV).
      
    Required Columns:
      - O_pred, C_pred, E_pred, A_pred, N_pred   [each in range 1..5]
      - TENURE_COUNT_MOS                         [numeric, can be float]
      - PAYMENT_CATEGORY_CODE                    [assumed numeric: e.g., 1=prepaid, 2=postpaid]
      - CLV_OVERALL_REVENUE                      [numeric, can be positive or negative]
    
    Outputs:
      - internal_raw, internal_z
      - tenure_z, payment_raw, payment_z
      - RS_z, RS_total
      - TUS
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the calculator with a copy of the DataFrame to avoid modifying the original.
        
        Args:
            df (pd.DataFrame): Input DataFrame with required columns.
        """
        self.df = df.copy()
        self.scaler_int = StandardScaler()
        self.scaler_tenure = StandardScaler()
        self.scaler_pay = StandardScaler()

    def compute_internal_score(self):
        """
        Computes the internal resistance score based on personality traits.
        
        Logic:
          - Higher Openness (O) & Agreeableness (A) => Lower friction (negative effect).
          - Higher Conscientiousness (C), Extraversion (E), and Neuroticism (N) => Higher friction.
        """
        self.df['internal_raw'] = (
              (6 - self.df['O_pred'])   # Higher O => lower friction
            + (6 - self.df['A_pred'])   # Higher A => lower friction
            + self.df['C_pred']         # Higher C => higher friction
            + self.df['E_pred']         # Higher E => higher friction
            + self.df['N_pred']         # Higher N => higher friction
        )

        # Standardize the internal resistance score
        self.df['internal_z'] = self.scaler_int.fit_transform(self.df[['internal_raw']])

    def compute_tenure_score(self):
        """
        Standardizes the tenure column (TENURE_COUNT_MOS).
        
        - Converts to numeric (handling errors).
        - Drops missing values to ensure clean data.
        - Standardizes using sklearn's StandardScaler.
        """
        self.df['TENURE_COUNT_MOS'] = pd.to_numeric(self.df['TENURE_COUNT_MOS'], errors='coerce')
        self.df.dropna(subset=['TENURE_COUNT_MOS'], inplace=True)
        self.df['tenure_z'] = self.scaler_tenure.fit_transform(self.df[['TENURE_COUNT_MOS']])

    def compute_payment_score(self):
        """
        Maps and standardizes the PAYMENT_CATEGORY_CODE:
        
        - If 2 (Postpaid) => less friction => numeric = 1.0
        - Otherwise (Prepaid/Other) => more friction => numeric = 3.0
        """
        self.df['payment_raw'] = self.df['PAYMENT_CATEGORY_CODE'].apply(lambda x: 1.0 if x == 2 else 3.0)
        self.df['payment_z'] = self.scaler_pay.fit_transform(self.df[['payment_raw']])

    def compute_resistance_score(self):
        """
        Computes the total resistance score (RS_total).
        
        Formula:
          RS_z = internal_z + payment_z - tenure_z
        
        Ensures RS_total is always positive by shifting values if necessary.
        """
        self.df['RS_z'] = self.df['internal_z'] + self.df['payment_z'] - self.df['tenure_z']

        # Ensure RS_z is always positive (to avoid division by zero or negative values)
        min_val = self.df['RS_z'].min()
        shift_amount = abs(min_val) + 0.001 if min_val <= 0 else 0
        self.df['RS_total'] = self.df['RS_z'] + shift_amount

    def compute_tus(self):
        """
        Computes the Transaction Utility Score (TUS):
        
        Formula:
          TUS = CLV_OVERALL_REVENUE / RS_total
        
        - Drops missing CLV values.
        - Handles cases where RS_total = 0 to avoid division errors.
        """
        self.df.dropna(subset=['CLV_OVERALL_REVENUE'], inplace=True)
        self.df['TUS'] = self.df.apply(
            lambda row: row['CLV_OVERALL_REVENUE'] / row['RS_total'] if row['RS_total'] != 0 else None,
            axis=1
        )

    def process(self):
        """
        Runs all computations in sequence.
        
        Returns:
            pd.DataFrame: DataFrame with additional computed columns.
        """
        self.compute_internal_score()
        self.compute_tenure_score()
        self.compute_payment_score()
        self.compute_resistance_score()
        self.compute_tus()
        return self.df


if __name__ == "__main__":
    import os

    # File path to data
    data_path = r'C:\Users\13032\Desktop\ph_behavior\big_5_personality\10k_predictions.csv'
    
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)

        # Initialize and process calculations
        calculator = ResistanceScoreCalculator(df_raw)
        df_result = calculator.process()

        # Display computed values
        print(df_result[[
            'CLV_OVERALL_REVENUE',
            'internal_raw', 'internal_z',
            'tenure_z', 'payment_raw', 'payment_z',
            'RS_total', 'TUS'
        ]].head())

        # Save results
        out_path = r'C:\Users\13032\Desktop\ph_behavior\big_5_personality\10k_predictions_with_tus_negative_possible.csv'
        df_result.to_csv(out_path, index=False)
    else:
        print(f"File not found: {data_path}")
