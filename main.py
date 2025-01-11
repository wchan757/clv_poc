import argparse
import os
from big_5_prediction import PersonalityPredictor
from clv_training import CLVTrainer
from group_assignment import GroupAssignment
from group_segmentation import GroupSegmentation
from resistance_score_calculation import ResistanceScoreCalculator
from result_parsing import ResultParser
from customer_segmentation import CustomerSegmentation  # Import the CustomerSegmentation class

def main(
    input_excel,
    model_folder,
    output_folder,
    big5_output_file="big_5_predictions.csv",
    clv_model_file="clv_model.pkl",
    group_assignment_file="group_assignments.csv",
    group_segmentation_file="group_segments.csv",
    resistance_scores_file="resistance_scores.csv",
    final_results_file="final_results.csv",
    customer_segmentation_output="customer_segmentation_results.csv"
):
    """
    Main function for executing the pipeline with dynamic parameters.

    Parameters:
    - input_excel: Path to the input Excel file.
    - model_folder: Path to the folder containing pre-trained models.
    - output_folder: Path to save all output files.
    - big5_output_file: Name of the file to save Big Five predictions.
    - clv_model_file: Name of the file to save the CLV model.
    - group_assignment_file: Name of the file to save group assignments.
    - group_segmentation_file: Name of the file to save group segments.
    - resistance_scores_file: Name of the file to save resistance scores.
    - final_results_file: Name of the file to save the final parsed results.
    - customer_segmentation_output: Name of the file to save customer segmentation results.
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Predict Big Five Personality Traits
    print("Step 1: Predicting Big Five Personality Traits...")
    predictor = PersonalityPredictor(
        input_excel=input_excel,
        model_folder=model_folder,
        output_csv=os.path.join(output_folder, big5_output_file)
    )
    predictor.run()

    # Step 2: Train Customer Lifetime Value (CLV) Models
    print("Step 2: Training CLV models...")
    clv_trainer = CLVTrainer(
        input_csv=os.path.join(output_folder, big5_output_file),
        output_model=os.path.join(output_folder, clv_model_file)
    )
    clv_trainer.run()

    # Step 3: Assign Groups Based on Predictions
    print("Step 3: Assigning groups...")
    group_assigner = GroupAssignment(
        input_csv=os.path.join(output_folder, big5_output_file),
        output_csv=os.path.join(output_folder, group_assignment_file)
    )
    group_assigner.run()

    # Step 4: Segment Groups for Analysis
    print("Step 4: Performing group segmentation...")
    segmenter = GroupSegmentation(
        input_csv=os.path.join(output_folder, group_assignment_file),
        output_csv=os.path.join(output_folder, group_segmentation_file)
    )
    segmenter.run()

    # Step 5: Calculate Resistance Scores
    print("Step 5: Calculating resistance scores...")
    resistance_calculator = ResistanceScoreCalculator(
        input_csv=os.path.join(output_folder, group_segmentation_file),
        output_csv=os.path.join(output_folder, resistance_scores_file)
    )
    resistance_calculator.run()

    # Step 6: Parse Results for Final Output
    print("Step 6: Parsing results...")
    parser = ResultParser(
        input_csv=os.path.join(output_folder, resistance_scores_file),
        output_csv=os.path.join(output_folder, final_results_file)
    )
    parser.run()

    # **Step 7: Customer Segmentation (New Step)**
    print("Step 7: Performing Customer Segmentation...")
    segmentation = CustomerSegmentation(
        file_path=os.path.join(output_folder, final_results_file)
    )
    segmentation.process(os.path.join(output_folder, customer_segmentation_output))

    print(f"Pipeline completed. Final segmentation results saved to {os.path.join(output_folder, customer_segmentation_output)}.")

if __name__ == "__main__":
    # Use argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Run the Personality Analysis Pipeline.")
    parser.add_argument("--input_excel", type=str, required=True, help="Path to the input Excel file.")
    parser.add_argument("--model_folder", type=str, required=True, help="Path to the folder containing pre-trained models.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save all output files.")
    parser.add_argument("--big5_output_file", type=str, default="big_5_predictions.csv", help="Name of the file to save Big Five predictions.")
    parser.add_argument("--clv_model_file", type=str, default="clv_model.pkl", help="Name of the file to save the CLV model.")
    parser.add_argument("--group_assignment_file", type=str, default="group_assignments.csv", help="Name of the file to save group assignments.")
    parser.add_argument("--group_segmentation_file", type=str, default="group_segments.csv", help="Name of the file to save group segments.")
    parser.add_argument("--resistance_scores_file", type=str, default="resistance_scores.csv", help="Name of the file to save resistance scores.")
    parser.add_argument("--final_results_file", type=str, default="final_results.csv", help="Name of the file to save the final parsed results.")
    parser.add_argument("--customer_segmentation_output", type=str, default="customer_segmentation_results.csv", help="Name of the file to save customer segmentation results.")

    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        input_excel=args.input_excel,
        model_folder=args.model_folder,
        output_folder=args.output_folder,
        big5_output_file=args.big5_output_file,
        clv_model_file=args.clv_model_file,
        group_assignment_file=args.group_assignment_file,
        group_segmentation_file=args.group_segmentation_file,
        resistance_scores_file=args.resistance_scores_file,
        final_results_file=args.final_results_file,
        customer_segmentation_output=args.customer_segmentation_output
    )
