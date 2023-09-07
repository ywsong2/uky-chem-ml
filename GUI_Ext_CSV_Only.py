"""
This program extends GUI.py to classify specific steps in a parent folder. 
"""

import os
import sys
from glob import glob
from argparse import ArgumentParser


# Argument parsing setup
parser = ArgumentParser(description='Graphical User Interface Extension for creating Result.csv only - (reprediciton purpose)')
parser.add_argument('-i','--inputfolder', metavar='', type=str, help='Input parent folder name', default='')
parser.add_argument('-s','--step', metavar='', type=str, help='Step(s) to check', default='')
args = parser.parse_args()

# Main
if __name__ == '__main__':
    # Pre-defined data structure
    list_of_all_steps = ['0','1','2','3','0m','1m','2m','3m','am']

    # Flags
    is_inputfolder_specified = True
    is_step_specified = True

    # Get the input arguments
    INPUT_DIR = args.inputfolder
    TARGET_STEPS = args.step.lower()

    # Check if input arguments are presented
    if not INPUT_DIR:
        is_inputfolder_specified = False
        print('ERROR: Please specify input folder!')
        sys.exit(0)
    
    if not TARGET_STEPS:
        is_step_specified = False
        print('Steps are not specified. Target all steps')

    # Get all sub folders in the target input folder that has Data.txt in it (Basically means all subfolders)
    target_folders = glob(INPUT_DIR + '/*/Data.txt')
    print('Number of target folders=', len(target_folders))

    # If there is no valid target folders, quit!
    if not target_folders:
        print('ERROR: The target folder does not have any valid sub folders!')
        sys.exit(0)

    for each_target_folder in target_folders:
        target_folder_name = os.path.dirname(each_target_folder)
        print('Processing: ', target_folder_name)
        if is_step_specified:
            print('python GUI_CSV_Only.py -i "' + target_folder_name + '" -s ' + TARGET_STEPS)
            os.system('python GUI_CSV_Only.py -i "' + target_folder_name + '" -s ' + TARGET_STEPS)
        else:
            print('python GUI_CSV_Only.py -i "' + target_folder_name + '"')
            os.system('python GUI_CSV_Only.py -i "' + target_folder_name + '"')