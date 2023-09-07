# Import library
import pandas as pd
import numpy as np
import glob
import os
import sys
from argparse import ArgumentParser

# Example of execution
# python ResultMerger.py -i target_dir_name -o Outfile_name.csv

# Argument parsing setup
parser = ArgumentParser(description="Merge results")
parser.add_argument(
    "-i",
    "--input",
    metavar="",
    type=str,
    help="Input folder name",
    default="input_folder",
)
parser.add_argument(
    "-o",
    "--output",
    metavar="",
    type=str,
    help="Output csv file name",
    default="Summary.csv",
)
parser.add_argument("-v", action="store_true")
args = parser.parse_args()

# Main
if __name__ == "__main__":
    # Get the input arguments
    INPUT_DIR = args.input
    OUTPUT_FILENAME = args.output
    OUTPUT_FILENAME = INPUT_DIR + "/" + OUTPUT_FILENAME
    DEBUG_MODE = False

    if args.v:
        DEBUG_MODE = True

    print("-- Configuration --")
    print("Debug mode =", DEBUG_MODE)
    print("Input folder =", INPUT_DIR)
    print("Output filename =", OUTPUT_FILENAME)

    # Prepare summary dictionary
    #   TNI: Total Number of Images
    #   CBP: Correct Binary Prediction
    #   CP: Correct Predictions (Binary + Step)
    #   BACC: Binary Accuracy
    #   OACC: Overall Accuracy
    #
    # TNI does not include maybe tagged ones
    summary_dict = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "0M": 0,
        "1M": 0,
        "2M": 0,
        "3M": 0,
        "TNI": 0,
        "CBP": 0,
        "CP": 0,
        "BACC": 0,
        "OACC": 0,
    }

    # CSV columns
    summary_columns = [
        "DIR_NAME",
        "0",
        "1",
        "2",
        "3",
        "0M",
        "1M",
        "2M",
        "3M",
        "TNI",
        "CBP",
        "CP",
        "BACC",
        "OACC",
    ]
    summary_df = pd.DataFrame(columns=summary_columns)
    zero_value_row = np.zeros(shape=(1, summary_df.shape[1]))

    # Counter
    valid_result_counter = 0

    # Go through all sub-directories in the input folder to get all Result.csv files
    for each_result in glob.iglob(INPUT_DIR + "/*/Result.csv"):
        # Print status
        print("Processing on -", each_result)

        valid_result_counter += 1

        # Create empty row
        empty_summury_row = pd.DataFrame(zero_value_row, columns=summary_columns)
        empty_summury_row.DIR_NAME = each_result

        # Open each result
        df = pd.read_csv(each_result)

        # Get number of each category (temporary dictionary is needed because to_dict() converts to int key when there is no 'M'(maybe) steps)
        value_count_dict_temp = df.UPDATED_STEP.value_counts().to_dict()

        # Declare empty dictionary
        value_count_dict = {}

        # Convert temp dictionary to another dictionary that use string as key
        for key, value in value_count_dict_temp.items():
            value_count_dict[str(key)] = value

        if DEBUG_MODE:
            print("Number of steps in the Result.csv =", value_count_dict)

        # Summurize
        # Get key, value pair from all category cases
        for key, value in summary_dict.items():
            if DEBUG_MODE:
                print("[", key, "]", "=", value)

            # Check if the target key is in the current Result.csv file
            if key in value_count_dict:
                # Update summary empty row
                empty_summury_row[key] = value_count_dict[key]
                # Add up
                summary_dict[key] += value_count_dict[key]

        # Get accuracy of prediction
        # Get number of rows
        num_of_rows = df.shape[0]

        # Get correct number of overall predictions
        correct_pred_counter = 0
        # Get correct number of binary predictions
        correct_binary_pred_counter = 0

        maybe_counter = 0
        for index, row in df.iterrows():
            # Count correct predictions
            if str(row.ORI_PRED_STEP) == str(row.UPDATED_STEP):
                correct_pred_counter += 1

            # Count correct binary predictions (0 == 0 or !0 == !0)
            if (str(row.ORI_PRED_STEP) == "0") and (str(row.UPDATED_STEP) == "0"):
                correct_binary_pred_counter += 1
            elif (str(row.ORI_PRED_STEP) != "0") and (str(row.UPDATED_STEP) != "0"):
                correct_binary_pred_counter += 1

            # Count number of maybe
            if "M" in str(row.UPDATED_STEP):
                maybe_counter += 1

        # Update summary dict (TNI, CP, BACC, OACC)
        summary_dict["TNI"] += num_of_rows - maybe_counter
        summary_dict["CP"] += correct_pred_counter
        summary_dict["CBP"] += correct_binary_pred_counter
        summary_dict["BACC"] = (summary_dict["CBP"] / summary_dict["TNI"]) * 100.0
        summary_dict["OACC"] = (summary_dict["CP"] / summary_dict["TNI"]) * 100.0

        # Update empty row (TNI, CP, OACC)
        empty_summury_row.TNI = num_of_rows - maybe_counter
        empty_summury_row.CP = correct_pred_counter
        empty_summury_row.CBP = correct_binary_pred_counter
        empty_summury_row.BACC = format(
            (correct_binary_pred_counter / (num_of_rows - maybe_counter)) * 100.0, ".4f"
        )
        empty_summury_row.OACC = format(
            (correct_pred_counter / (num_of_rows - maybe_counter)) * 100.0, ".4f"
        )

        # Append / Sort / Export
        # summary_df = summary_df.append(empty_summury_row)
        summary_df = pd.concat([summary_df, empty_summury_row], ignore_index=True)

        if DEBUG_MODE:
            print("Summray =", summary_dict)
            break

    # Create empty row and save all summary dictionary information into output CSV file
    empty_summury_row = pd.DataFrame(zero_value_row, columns=summary_columns)
    empty_summury_row["DIR_NAME"] = "Summary"
    empty_summury_row["0"] = summary_dict["0"]
    empty_summury_row["1"] = summary_dict["1"]
    empty_summury_row["2"] = summary_dict["2"]
    empty_summury_row["3"] = summary_dict["3"]
    empty_summury_row["0M"] = summary_dict["0M"]
    empty_summury_row["1M"] = summary_dict["1M"]
    empty_summury_row["2M"] = summary_dict["2M"]
    empty_summury_row["3M"] = summary_dict["3M"]
    empty_summury_row.TNI = summary_dict["TNI"]
    empty_summury_row.CP = summary_dict["CP"]
    empty_summury_row.CBP = summary_dict["CBP"]
    empty_summury_row.BACC = format(summary_dict["BACC"], ".4f")
    empty_summury_row.OACC = format(summary_dict["OACC"], ".4f")
    # summary_df = summary_df.append(empty_summury_row)
    summary_df = pd.concat([summary_df, empty_summury_row], ignore_index=True)
    summary_df.to_csv(OUTPUT_FILENAME, encoding="utf-8", index=False)
