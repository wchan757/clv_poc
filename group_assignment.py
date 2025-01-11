import pandas as pd

class PercentileTUSClustering:
    """
    A class to segment users based on TUS using percentile-based binning 
    for a balanced distribution, while adding a descriptive group name.
    """

    def __init__(self):
        """
        Initializes the clustering model.
        """
        pass

    def fit_cluster(self, df):
        """
        Segments users into percentiles based on TUS and adds a descriptive group name.

        Parameters:
        - df: DataFrame with 'TUS' column.

        Returns:
        - df: DataFrame with new 'TUS_Cluster' and 'TUS_Group_Name' columns.
        """
        df = df.dropna(subset=['TUS']).copy()

        # Compute TUS percentiles
        df['TUS_percentile'] = df['TUS'].rank(pct=True)

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
# Example Usage
# ------------------------------------------------------
if __name__ == "__main__":
    input_file = r"C:\Users\13032\Desktop\ph_behavior\big_5_personality\10k_predictions_with_tus_negative_possible_wthoup.csv"
    output_file = r"C:\Users\13032\Desktop\ph_behavior\big_5_personality\10k_predictions_with_tus_negative_possible_with_group.csv"

    df_raw = pd.read_csv(input_file)

    # Initialize the improved clustering model
    tus_clustering = PercentileTUSClustering()

    # Perform Clustering
    df_clustered = tus_clustering.fit_cluster(df_raw)

    # Display sample output
    print(df_clustered[['TUS', 'TUS_percentile', 'TUS_Cluster', 'TUS_Group_Name']].head(20))

    # Save results
    tus_clustering.save_results(df_clustered, output_file)

    # Check counts of each group
    print(df_clustered['TUS_Cluster'].value_counts(normalize=True) * 100)  # Show % instead of raw counts
