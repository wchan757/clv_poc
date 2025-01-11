import os
import pandas as pd
from poe_api_wrapper import PoeApi

class PersonalityAnalysis:
    def __init__(self, input_file: str, output_folder: str, api_tokens: dict, bot_name: str = "gpt4_o_128k", batch_size: int = 200):
        """
        Initialize the PersonalityAnalysis class.
        
        :param input_file: Path to the input Excel file.
        :param output_folder: Directory where output files will be saved.
        :param api_tokens: Dictionary containing API tokens.
        :param bot_name: The name of the bot to be used.
        :param batch_size: Number of rows per batch for processing.
        """
        self.input_file = input_file
        self.output_folder = output_folder
        self.api_tokens = api_tokens
        self.bot_name = bot_name
        self.batch_size = batch_size
        self.client = PoeApi(tokens=api_tokens)
        
        self._ensure_output_directory()
        self.df_part = self._load_data()
    
    def _ensure_output_directory(self):
        """Ensure the output folder exists."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def _load_data(self):
        """Load and sort the input dataset."""
        df = pd.read_excel(self.input_file)
        return df.sort_values(by='SUBSCRIBER_ID').reset_index(drop=True)
    
    def _send_message_to_poe(self, prompt: str):
        """Send a message to the Poe bot and return the response."""
        try:
            response_text = ""
            for chunk in self.client.send_message(self.bot_name, prompt):
                response_text += chunk["response"]
                print(chunk["response"], end="", flush=True)
            return response_text
        except Exception as e:
            print(f"Error during API call: {e}")
            return None
    
    def process_batches(self):
        """Process the data in batches and send it to the bot."""
        num_batches = len(self.df_part) // self.batch_size + (1 if len(self.df_part) % self.batch_size > 0 else 0)
        prompt_template = (
            """
            Now you are a psychologist.
            For each subscriber, return only the following fields in a tabular format:
            1. SUBSCRIBER_ID
            2. Big 5 Personality Traits (O, C, E, A, N) on a scale from 1 to 5.
            Based on the information I provide you, return all results in a table format without asking.
            
            Here is the data:
            """
        )

        for batch_num in range(num_batches):
            print(f"Processing batch {batch_num + 1} of {num_batches}...")
            start_idx = batch_num * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_df = self.df_part.iloc[start_idx:end_idx]

            column_headers = " | ".join(batch_df.columns)
            data_rows = "\n".join([
                " | ".join([str(row[col]) if pd.notnull(row[col]) else 'NA' for col in batch_df.columns])
                for _, row in batch_df.iterrows()
            ])
            table = f"{column_headers}\n{data_rows}"
            prompt = prompt_template + table

            bot_response = self._send_message_to_poe(prompt)
            if bot_response:
                self._save_response(bot_response, batch_num)
            else:
                print(f"No response received for batch {batch_num + 1}.")
        
        print("Processing completed.")
    
    def _save_response(self, response: str, batch_num: int):
        """Save the bot's response to a text file."""
        output_file = os.path.join(self.output_folder, f'cinoketext_{batch_num + 1}.txt')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Batch {batch_num + 1} response saved to {output_file}")
        except Exception as e:
            print(f"Error saving response for batch {batch_num + 1}: {e}")

if __name__ == "__main__":
    # Example usage
    tokens = {
        'p-b': os.getenv('POE_API_TOKEN_BASIC', 'your_default_token'),
        'p-lat': os.getenv('POE_API_TOKEN_LAT', 'your_default_token')
    }
    
    input_path = r'C:\Users\13032\Desktop\ph_behavior\big_5_personality\10k_sampling_noMSISDN.xlsx'
    output_path = r'C:\Users\13032\Desktop\ph_behavior\participant_response'
    
    analyzer = PersonalityAnalysis(input_file=input_path, output_folder=output_path, api_tokens=tokens)
    analyzer.process_batches()
