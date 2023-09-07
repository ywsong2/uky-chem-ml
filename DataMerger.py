# Import library
import pandas as pd
import numpy as np
import glob
import os
import sys
from argparse import ArgumentParser

# Example of execution
# python DataMerger.py -i target_dir_name -o Outfile_name.txt 

# Argument parsing setup
parser = ArgumentParser(description='Merge data')
parser.add_argument('-i','--input', metavar='', type=str, help='Input folder name', default='input_folder')
parser.add_argument('-o','--output', metavar='', type=str, help='Output data file name', default='MergedData.txt')
parser.add_argument('-v',action='store_true')
args = parser.parse_args()

# Main
if __name__ == '__main__':
    # Get the input arguments
    INPUT_DIR = args.input
    OUTPUT_FILENAME = args.output
    OUTPUT_FILENAME = INPUT_DIR + '/' + OUTPUT_FILENAME
    DEBUG_MODE = False

    if args.v:
        DEBUG_MODE = True

    print('-- Configuration --')
    print('Debug mode =', DEBUG_MODE)
    print('Input folder =', INPUT_DIR)
    print('Output filename =', OUTPUT_FILENAME)

    # Merged data file
    merged_data_fp = open(OUTPUT_FILENAME, 'w')
    
    # Counter
    valid_data_file_counter = 0
    valid_data_counter = 0
    skip_data_counter = 0

    # Go through all sub-directories in the input folder to get all Data.txt files
    for each_data in glob.iglob(INPUT_DIR + '/*/Data.txt'):
        # Open Result.csv 
        result_df = pd.read_csv(os.path.dirname(each_data) + '/Result.csv')

        # Check if data frame is empty
        if result_df.empty:
            print('ERROR: Result.csv is empty!')
            continue
        
        # Open data file
        with open(each_data) as fp:
            # Print status
            print('Processing on -', each_data)

            # Increase validness counter
            valid_data_file_counter += 1

            # Read lines
            lines = fp.readlines()

            # Loop through all data's line
            for each_line in lines:
                # Check if data contains 'nan' (if so, skip)
                if 'nan' in each_line:
                    continue

                # Split data 
                splited_line_data = each_line.split()
                
                # Check validness of data (just in case)
                if len(splited_line_data) != 801:
                    print('Error: Invalid input data (length of splited string is not 801)')
                    continue
                    
                # Get the ID from data
                data_ID = splited_line_data.pop(0)
                
                # Concatanate data
                each_data = " ".join(splited_line_data)
                
                # Get the UPDATED_STEP
                result_row = result_df.loc[result_df.ID == int(data_ID)]
                updated_step = str(result_row.UPDATED_STEP.values[0])
                
                #print('updated step = ', updated_step)

                # Pick only clear classifications
                if updated_step in ['0', '1', '2', '3']:
                    #print('increasing valid counter')
                    valid_data_counter += 1
                    merged_data_fp.write(updated_step + ' ' + each_data + '\n')
                else:
                    # skip_data_counter += 1
                    # Instead of skipping maybe tagged ones, add them as 0 step
                    valid_data_counter += 1
                    merged_data_fp.write('0 ' + each_data + '\n')

    # Close merged data file
    merged_data_fp.close()

    print('Total number of merged data files=', valid_data_file_counter)
    print('Total number of merged data =', valid_data_counter)
    print('Total number of skipped data =', skip_data_counter)


        
        

       

