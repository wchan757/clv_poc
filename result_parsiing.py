import os
import pandas as pd
import re

class TextDataProcessor:
    def __init__(self, folder_path="participant_response", file_prefix="cinoketext_", output_csv="combined_data.csv", join_file=None, join_key="SUBSCRIBER_ID"):
        """
        Initialize the TextDataProcessor class.

        :param folder_path: Path to the folder containing text files.
        :param file_prefix: Prefix of the files to process (e.g., 'cinoketext_').
        :param output_csv: Name of the output CSV file.
        :param join_file: Path to the CSV or Excel file to join with.
        :param join_key: The key column to perform an inner join.
        """
        self.folder_path = folder_path
        self.file_prefix = file_prefix
        self.output_csv = output_csv
        self.join_file = join_file
        self.join_key = join_key
        self.data = []

    def process_files(self):
        """Reads and processes all text files matching 'cinoketext_*.txt' in the specified folder."""
        print(f"Processing text files in folder: {self.folder_path}")

        # Check if the folder exists
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder '{self.folder_path}' not found.")

        # List all files in the folder
        for file_name in os.listdir(self.folder_path):
            if file_name.startswith(self.file_prefix) and file_name.endswith(".txt"):  # Match files like 'cinoketext_1.txt'
                file_path = os.path.join(self.folder_path, file_name)

                print(f"Processing file: {file_name}")

                # Read and process the file
                with open(file_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()

                    for line in lines:
                        # Skip headers, separators, and empty lines
                        if "SUBSCRIBER_ID" in line or "---" in line or not line.strip():
                            continue

                        # Split the line into columns based on the pipe '|' delimiter
                        columns = [col.strip() for col in line.split("|") if col.strip()]

                        # Append the processed row to the data list
                        if len(columns) == 6:  # Ensure it has all required columns
                            self.data.append(columns)

        print("Processing complete.")

    def save_to_csv(self):
        """Converts the processed data into a Pandas DataFrame and saves it as a CSV file."""
        if not self.data:
            print("No data found to save.")
            return

        df = pd.DataFrame(self.data, columns=["SUBSCRIBER_ID", "O", "C", "E", "A", "N"])

        # If join_file is provided, perform an inner join
        if self.join_file:
            if self.join_file.endswith(".csv"):
                df_join = pd.read_csv(self.join_file)
            elif self.join_file.endswith(".xlsx"):
                df_join = pd.read_excel(self.join_file)
            else:
                raise ValueError("Unsupported file format for joining.")

            df = df.merge(df_join, on=self.join_key, how="inner")

        df.to_csv(self.output_csv, index=False, encoding="utf-8")
        print(f"Combined data has been saved to {self.output_csv}")

    def run(self):
        """Executes the full data processing pipeline."""
        self.process_files()
        self.save_to_csv()


# Example usage
if __name__ == "__main__":
    processor = TextDataProcessor(
        folder_path="participant_response", 
        file_prefix="cinoketext_", 
        output_csv="combined_data.csv", 
        join_file="join_data.csv",  # Specify the file to join
        join_key="SUBSCRIBER_ID"
    )
    processor.run()
