import os
import glob
import sys
import pandas
import numpy as np
from PIL import Image, ImageTk
from argparse import ArgumentParser
from tkinter import filedialog

# Argument parsing setup
parser = ArgumentParser(description='Graphical User Interface for generating Result.csv only')
parser.add_argument('-i','--inputfolder', metavar='', type=str, help='Input folder name', default='')
parser.add_argument('-s','--step', metavar='', type=str, help='Step(s) to check', default='')
args = parser.parse_args()

# Main
if __name__ == '__main__':
    # Pre-defined data structure
    # am: All M steps
    # nz: All non-zero steps (1,2,3)
    list_of_all_steps = ['0','1','2','3','0m','1m','2m','3m','am','nz']
    list_of_target_steps = []

    # Flags
    is_inputfolder_specified = True
    is_step_specified = True

    # Get the input arguments
    INPUT_DIR = args.inputfolder
    TARGET_STEPS = args.step.lower()

    # Check if input arguments are presented
    if not INPUT_DIR:
        is_inputfolder_specified = False
    
    if not TARGET_STEPS:
        is_step_specified = False

    # Print out configuration
    print('\n\n---Configurations---')
    if is_inputfolder_specified:
        print('Input folder (specified) =', INPUT_DIR)
    else:
        print('Input folder =', INPUT_DIR)
    if is_step_specified:
        print('Target steps (specified) =', TARGET_STEPS)
    else:
        print('Target steps = All steps')
        list_of_target_steps = list_of_all_steps
    
    # Check if the specified step is valid
    if TARGET_STEPS:
        if TARGET_STEPS not in list_of_all_steps:
            print('ERROR: Specified step is invalid. Please check your target step.')
            print('Exiting software...')
            sys.exit(0)
        else:
            if TARGET_STEPS == '0':
                list_of_target_steps.append('0')
            elif TARGET_STEPS == '1':
                list_of_target_steps.append('1')
            elif TARGET_STEPS == '2':
                list_of_target_steps.append('2')
            elif TARGET_STEPS == '3':
                list_of_target_steps.append('3')
            elif TARGET_STEPS == '0m':
                list_of_target_steps.append('0M')
            elif TARGET_STEPS == '1m':
                list_of_target_steps.append('1M')
            elif TARGET_STEPS == '2m':
                list_of_target_steps.append('2M')
            elif TARGET_STEPS == '3m':
                list_of_target_steps.append('3M')
            elif TARGET_STEPS == 'am':
                list_of_target_steps.append('0M')
                list_of_target_steps.append('1M')
                list_of_target_steps.append('2M')
                list_of_target_steps.append('3M')
            else:
                print('ERROR: Invalid step!')

        print('The list of target steps = ', list_of_target_steps)
        
    # Create tkinter pointer
    # tk_root = tk.Tk()
    # tk_root.title("ML Classification Viewer")

    # Global variables
    # current_win_size = 3    # Default windows size = 3
    img_list = []               # Image path list
    img_counter = 0             # How many images we have?
    img_dict = {}               # Image dictionary { image id : steps }
    # img_index = -1              # Image index (use to access image directly): Start with -1
    # img_path = ""               # Current image path
    # target_folder  = ""         # Target folder
    # isDone = False              # Done?
    # isDebug = False             # Debug?
    # tk_label_prediction = 0     # Declearation 
    # tk_label_total = 0          # Declearation
    # tk_pic = 0                  # Declearation

    # Get image name (based on img_index)
    def get_img_path(folder, goback=False):
        global img_index
        global img_counter
        global isDone

        if goback == True:
            # Decrease img_index by 1
            img_index -= 1
            # Make it sure img_index is at least bigger than 0
            if img_index > 0:
                # Keep going back until it finds target steps
                while img_dict[str(img_index)] not in list_of_target_steps:
                    # Decrease current image index
                    img_index -= 1
                    # Check if it is equal to 0
                    if img_index <= 0:
                        break
            else:
                print('Warning: Cannot go back anymore!')

        else:
            # Increase img index 
            img_index += 1
            # Make it sure img_index is less than img_counter (total number of images)
            if img_index < img_counter:
                while img_dict[str(img_index)] not in list_of_target_steps:
                    # Increase current image index
                    img_index += 1
                    # Check if all done
                    if img_index >= (img_counter-1):
                        break
            else:
                print('Warning: Cannot go forward anymore!')

        # Check min value
        if img_index < 0:
            img_index = 0

        # Check max value
        if img_index >= img_counter:
            img_index = img_counter - 1

        # Create img file name
        img_path = folder + '/' + img_dict[str(img_index)] + '/' + str(img_index) + '.png'

        # Return img file name
        return img_path

    # Save result to CSV
    def save_result(dataframe):
        dataframe = dataframe.sort_values(by='ID')
        dataframe = dataframe.reset_index(drop=True)
        dataframe.to_csv(target_folder + '/Result.csv', encoding='utf-8', index=False)

    # # Save current dictionary to Result.CSV
    # def save_dict():
    #     global img_dict
        
    #     # If there is no Results.CSV, then make one
    #     if not os.path.isfile(target_folder + '/Result.csv'):
    #         # Create CSV file (result)
    #         columns = ['ID', 'ORI_PRED_STEP', 'UPDATED_STEP']
    #         df = pandas.DataFrame(columns=columns)
    #         zero_value_row = np.zeros(shape=(1, df.shape[1]))

    #         # Loop through dictionary and create initial CSV file
    #         for i in range(0, img_counter):
    #             orginal_predicted_step = img_dict[str(i)]
                
    #             # Create empty row 
    #             empty_row = pandas.DataFrame(zero_value_row, columns=columns)

    #             # Update value in the new empty row
    #             empty_row.ID = i
    #             empty_row.ORI_PRED_STEP = orginal_predicted_step
    #             empty_row.UPDATED_STEP = orginal_predicted_step

    #             # Append / Sort / Export
    #             df = df.append(empty_row)
    #             df = df.sort_values(by='ID')
    #             df = df.reset_index(drop=True)
    #             df.to_csv(target_folder + '/Result.csv', encoding='utf-8', index=False)
    #     else:
    #         # If the Results.CSV file already exists, update it
    #         # Open Result.csv file
    #         df = pandas.read_csv(target_folder + '/Result.csv')

    #         # Loop through data frame row and update value based on dictionary
    #         for index, row in df.iterrows():
    #             df.loc[index, 'UPDATED_STEP'] = img_dict[str(row.ID)]

    #         # Append / Sort / Export
    #         df = df.sort_values(by='ID')
    #         df = df.reset_index(drop=True)
    #         df.to_csv(target_folder + '/Result.csv', encoding='utf-8', index=False)

    # # Save CSV and quit 
    # def save_and_quit():
    #     save_dict()


    # If input folder is not specified, then ask user to select target folder
    if not is_inputfolder_specified:
        # First select target directory if not specified
        print('Please select target directory')
        exit(0)
    else:
        # If input folder is specified, use it as target folder
        target_folder = INPUT_DIR

    # Read all img names and prepare dictionary 
    for each_img in glob.iglob(target_folder + '/*/*.png'):
        # Check if the folder name is 'NZ' and if so, skip that case
        if 'NZ' in each_img:
            continue

        # Increase img_counter
        img_counter += 1
        img_list.append(each_img)

        # Check if the path is valid (it should be at least 3 level dir tree)
        if len(each_img.split(os.sep)) < 2:
            print('Error: Invalid image path!')
        else:
            # Split out path
            img_path_split = each_img.split(os.sep)
            # Img id is base name of path without extension 
            img_id = img_path_split[-1].split('.')[0]
            # Img step is name of parent folder 
            img_step = img_path_split[-2]
            # Set dictionary
            img_dict[img_id] = img_step

    # Check if there are any images (if not, quit program with error msg)
    if len(img_list) == 0:
        print('Error - The target folder', target_folder, 'does not have any images to classify')
        sys.exit(0)

    print("Total image:", img_counter)

    # Check if there is already Result.csv file (previously updated)
    if not os.path.isfile(target_folder + '/Result.csv'):
        # Create CSV file (result)
        columns = ['ID', 'ORI_PRED_STEP', 'UPDATED_STEP']
        df = pandas.DataFrame(columns=columns)
        zero_value_row = np.zeros(shape=(1, df.shape[1]))

        # Loop through dictionary and create initial CSV file
        for i in range(0, img_counter):
            orginal_predicted_step = img_dict[str(i)]
            
            # Create empty row 
            empty_row = pandas.DataFrame(zero_value_row, columns=columns)

            # Update value in the new empty row
            empty_row.ID = i
            empty_row.ORI_PRED_STEP = orginal_predicted_step
            empty_row.UPDATED_STEP = orginal_predicted_step

            # Append
            df = df.append(empty_row)

        # Sort and save
        print("Create and save Result.csv at", target_folder)
        save_result(df)
    else:
        print("Result.csv is already existing. Exiting...")

