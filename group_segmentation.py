import pandas as pd
import argparse
import os

class PercentileTUSClustering:
    """
    A class to segment users based on TUS using percentile-based binning 
    for a balanced distribution, while adding a descriptive group name.
    """

    def __init__(self, tus_col: str = "TUS"):
        """
        Initializes the clustering model.

        Parameters:
        - tus_col: Column name for TUS. Defaults to 'TUS'.
        """
        self.tus_col = tus_col

    def fit_cluster(self, df):
        """
        Segments users into percentiles based on TUS and adds a descriptive group name.

        Parameters:
        - df: DataFrame with the TUS column.

        Returns:
        - df: DataFrame with new 'TUS_Cluster' and 'TUS_Group_Name' columns.
        """
        if self.tus_col not in df.columns:
            raise ValueError(f"Error: Column '{self.tus_col}' not found in the dataset.")

        df = df.dropna(subset=[self.tus_col]).copy()

        # Compute TUS percentiles
        df['TUS_percentile'] = df[self.tus_col].rank(pct=True)

        # Define percentile-based binning
        def assign_priority(pct):
            if pct <= 0.25:
                return 'LOW_PRIORITY'
            elif pct <= 0.60:
                return 'MEDIUM_PRIORITY'
            elif pct <= 0.85:
                return 'HIGH_PRIORITY'
            else:
                return 'VERY_HIGH_PRIORITY'

        df['TUS_Cluster'] = df['TUS_percentile'].apply(assign_priority)

        # Map to human-readable group names
        group_names = {
            "LOW_PRIORITY": "Low Potential - High Resistance",
            "MEDIUM_PRIORITY": "Moderate Potential - Some Resistance",
            "HIGH_PRIORITY": "High Potential - Low Resistance",
            "VERY_HIGH_PRIORITY": "Top Potential - Very Low Resistance"
        }

        df['TUS_Group_Name'] = df['TUS_Cluster'].map(group_names)

        return df

    def save_results(self, df, output_path):
        """
        Save the processed DataFrame to a CSV file.

        Parameters:
        - df: Processed DataFrame with TUS Clusters and Group Names.
        - output_path: File path to save the results.
        """
        df.to_csv(output_path, index=False)
        print(f"Clustered data saved to {output_path}")

# ------------------------------------------------------
# Main Execution with Parameterization
# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment users into TUS clusters based on percentile binning.")

    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the processed output CSV file.")
    parser.add_argument("-t", "--tus_col", type=str, default="TUS", 
                        help="Column name for TUS. Defaults to 'TUS'.")
    parser.add_argument("--show", action="store_true", help="Display a sample of results.")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
    else:
        df_raw = pd.read_csv(args.input)

        # Initialize clustering with optional TUS column
        tus_clustering = PercentileTUSClustering(args.tus_col)

        # Perform Clustering
        try:
            df_clustered = tus_clustering.fit_cluster(df_raw)

            # Save results
            tus_clustering.save_results(df_clustered, args.output)

            # Optionally display sample output
            if args.show:
                print(df_clustered[[args.tus_col, 'TUS_percentile', 'TUS_Cluster', 'TUS_Group_Name']].head(20))
                print("\nCluster distribution (%):")
                print(df_clustered['TUS_Cluster'].value_counts(normalize=True) * 100)

        except ValueError as e:
            print(e)
