import os
import glob
import sys
import pandas
import tkinter as tk
import numpy as np
from tkinter import Frame, Button
from PIL import Image, ImageTk
from tkinter import filedialog
from argparse import ArgumentParser

# Argument parsing setup
parser = ArgumentParser(description='Graphical User Interface for SMCF')
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
    tk_root = tk.Tk()
    tk_root.title("ML Classification Viewer")

    # Global variables
    current_win_size = 3    # Default windows size = 3
    img_list = []               # Image path list
    img_counter = 0             # How many images we have?
    img_dict = {}               # Image dictionary { image id : steps }
    img_index = -1              # Image index (use to access image directly): Start with -1
    img_path = ""               # Current image path
    target_folder  = ""         # Target folder
    isDone = False              # Done?
    isDebug = False             # Debug?
    tk_label_prediction = 0     # Declearation 
    tk_label_total = 0          # Declearation
    tk_pic = 0                  # Declearation

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

    # Move to next image
    def next_image(goback=False):
        # Get global variables
        global img_path
        global current_win_size
        global target_folder
        global isDone

        # Reset current windows size to default size before show next image
        current_win_size = 3

        # Get photo path
        img_path = get_img_path(target_folder, goback)

        # Reset the background of label to white
        tk_label_prediction.config(bg='white')

        
        # As long as it is not going back, show 'done' when it is done. Otherwise, move forward
        if goback == False:
            if isDone:
                return
        # If this is go back, set isDone back to False
        else:
            isDone = False

        # Update prediction label
        tk_label_prediction.config(text=str(img_index) + '.png : ' + img_dict[str(img_index)] + ' step')
        tk_label_total.config(text='Progress: ' + str(img_index) + '/' + str(img_counter-1))


        # Show image on image gui
        photo = ImageTk.PhotoImage(Image.open(img_path))
        tk_pic.configure(image=photo, text='')
        tk_pic.image = photo

        # Check if all done
        if img_index >= (img_counter-1):
            isDone = True
            print('ALL DONE')
            # If it is all done, show All Done text on the text box
            tk_label_total.config(text='Progress: All Done!')
        else:
            isDone = False

    # Move file to updated step's folder
    def move_file(prev_step, updated_step):

        global target_folder
        global isDebug

        source_dir = target_folder + '/' + prev_step
        target_dir = target_folder + '/' + updated_step
        if isDebug:
            print('Debug: move_file()', source_dir, target_dir)

        # Create target folder if it does not exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        src_file = source_dir + '/' + str(img_index) + '.png'
        dst_file = target_dir + '/' + str(img_index) + '.png'
        if isDebug:
            print('Debug: move_file()', src_file, dst_file)
        
        os.rename(src_file, dst_file)

    # If ML predicted it correctly, then do not need to move files
    def yes():
        # Everything looks good, keep moving!
        next_image()

    def skip():
        # Do nothing and skip! (same as yes)
        next_image()

    def goback():
        # Somehow I need to go back 
        next_image(goback=True)

    # Left arrow key press handler
    def leftKey(event):
        # When users press the left arrow key, it should show smaller window size image
        # Get current_win_size
        global current_win_size
        global img_index

        # Make is sure it is not less than 0
        target_win_size = current_win_size - 1
        if target_win_size < 1:
            return
        else:
            current_win_size = current_win_size - 1
            target_win_size = str(target_win_size)
            
            # Target image filename
            target_img = target_folder + '/MultiWinSize/' + target_win_size + '/' + str(img_index) + '.png'
            
            # Print out current bin size
            print('Current Bin Size = ', target_win_size)

            # Update current image
            photo = ImageTk.PhotoImage(Image.open(target_img))
            tk_pic.configure(image=photo)
            tk_pic.image = photo

    # Right arrow key press handler
    def rightKey(event):
        # When users press the left arrow key, it should show larger window size image
        # Get current_win_size
        global current_win_size
        global img_index

        # Make is sure it is not larger than 5
        target_win_size = current_win_size + 1
        if target_win_size > 5:
            return
        else:
            current_win_size = current_win_size + 1
            target_win_size = str(target_win_size)
            
            # Target image filename
            target_img = target_folder + '/MultiWinSize/' + target_win_size + '/' + str(img_index) + '.png'
            
            # Print out current bin size
            print('Current Bin Size = ', target_win_size)

            # Update current image
            photo = ImageTk.PhotoImage(Image.open(target_img))
            tk_pic.configure(image=photo)
            tk_pic.image = photo

    # Refresh current image 
    def refresh_pred_label():
        # Get global variables
        global img_path
        global current_win_size
        global target_folder
        global isDone

        # Update prediction label
        tk_label_prediction.config(text=str(img_index) + '.png : ' + img_dict[str(img_index)] + ' step')

    # Handle enter key (which means OK)
    def enterKey(event):
        # If it is done, return right away
        if isDone:
            return

        next_image()

    # Handle number keyboard input (0 - 3)
    def numKey(event):
        global img_index

        # Label
        updated_label = event.char

        # Check if pressed key is pressed with shift key (shift+0 ~ shift+5)
        if event.char == ')':
            event.char = '0'
            updated_label = '0M'
        elif event.char == '!':
            event.char = '1'
            updated_label = '1M'
        elif event.char == '@':
            event.char = '2'
            updated_label = '2M'
        elif event.char == '#':
            event.char = '3'
            updated_label = '3M'

        # Debug info
        if True:
            print('Number key [', updated_label, '] pressed. Image id = ', img_index)

        if isDebug:
            print('Debug: numKey() Current pred =', img_dict[str(img_index)], 'Updated pred =', updated_label)
        
        # Move image to proper folder (if the step has been changed)
        if updated_label != img_dict[str(img_index)]:
            if isDebug:
                print('Debug: numKey() Step has been changed. Move file!')
            move_file(img_dict[str(img_index)], updated_label)
            # When the label has been changed, change color of label box
            tk_label_prediction.config(bg='yellow')
        else:
            if isDebug:
                print('Do not need to move')

        # After moving the image file, we need to update dictionary
        img_dict[str(img_index)] = updated_label

        # Update prediction label
        refresh_pred_label()

    # Save result to CSV
    def save_result(dataframe):
        dataframe = dataframe.sort_values(by='ID')
        dataframe = dataframe.reset_index(drop=True)
        dataframe.to_csv(target_folder + '/Result.csv', encoding='utf-8', index=False)

    # Save current dictionary to Result.CSV
    def save_dict():
        global img_dict
        
        # If there is no Results.CSV, then make one
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

                # Append / Sort / Export
                df = df.append(empty_row)
                df = df.sort_values(by='ID')
                df = df.reset_index(drop=True)
                df.to_csv(target_folder + '/Result.csv', encoding='utf-8', index=False)
        else:
            # If the Results.CSV file already exists, update it
            # Open Result.csv file
            df = pandas.read_csv(target_folder + '/Result.csv')

            # Loop through data frame row and update value based on dictionary
            for index, row in df.iterrows():
                df.loc[index, 'UPDATED_STEP'] = img_dict[str(row.ID)]

            # Append / Sort / Export
            df = df.sort_values(by='ID')
            df = df.reset_index(drop=True)
            df.to_csv(target_folder + '/Result.csv', encoding='utf-8', index=False)

    # Handle exiting
    def exitKey(event):
        save_and_quit()

    # Save CSV and quit 
    def save_and_quit():
        save_dict()
        tk_root.destroy()

    # If input folder is not specified, then ask user to select target folder
    if not is_inputfolder_specified:
        # First select target directory if not specified
        print('Please select target directory')
        target_folder = filedialog.askdirectory(initialdir='.')
        print('Target directory = ', target_folder)
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
            # df = df.append(empty_row)
            df = df.concat([df, empty_row])

        # Sort and save
        save_result(df)

    # Open Result.csv file
    df = pandas.read_csv(target_folder + '/Result.csv')

    # Start to layout GUI
    top_frame = Frame(tk_root)
    bottom_frame = Frame(tk_root)
    top_frame.pack(side='top')

    # Get first image filename
    img_path = get_img_path(target_folder)

    # Image GUI
    photo = ImageTk.PhotoImage(Image.open(img_path))
    tk_pic = tk.Label(tk_root, image=photo)
    tk_pic.image = photo
    tk_pic.pack(side='top')

    # Get prediction label
    pred_label_text = str(img_index) + '.png : ' + img_dict[str(img_index)] + ' step'
    tk_label_prediction = tk.Label(top_frame, text=pred_label_text, height=1)
    tk_label_prediction.config(font = ('times', 50, 'bold'))
    tk_label_prediction.pack(fill=tk.BOTH, pady=20)

    # Set total number of images in the target folder
    tk_label_total = tk.Label(top_frame, text='Progress: ' + str(img_index) + '/' + str(img_counter-1))
    tk_label_total.config(font = ('times', 20))
    tk_label_total.pack(side='top', pady=5)

    # Show the current file name
    tk_label_filename = tk.Label(top_frame, text='Current folder: ' + os.path.dirname(target_folder))
    tk_label_filename.config(font = ('times', 20))
    tk_label_filename.pack(side='top', pady = 3)

    # Put buttons
    tk_button_yes = Button(bottom_frame, fg='green', text="Yes", command=yes).pack(side='left', fill='x', expand='yes')
    tk_button_skip = Button(bottom_frame, text='Skip', command=skip).pack(side='left', fill='x', expand='yes')
    tk_button_goback = Button(bottom_frame, text='Go Back', command=goback).pack(side='left', fill='x', expand='yes')
    tk_button_quit = Button(bottom_frame, fg='red', text='Save and Quit', command=lambda: save_and_quit()).pack(side='left', fill='x', expand='yes')
    bottom_frame.pack(side='bottom', fill='both', expand='yes', pady=15, padx=20)

    # Bind keyboard
    tk_root.bind('<Left>', leftKey)
    tk_root.bind('<Right>', rightKey)
    tk_root.bind('<Return>', enterKey)
    #tk_root.bind('<Key>', numKey) => Maybe better option (but let'e keep it simple)
    tk_root.bind(')', numKey)
    tk_root.bind('0', numKey)
    tk_root.bind('!', numKey)
    tk_root.bind('1', numKey)
    tk_root.bind('@', numKey)
    tk_root.bind('2', numKey)
    tk_root.bind('#', numKey)
    tk_root.bind('3', numKey)
    tk_root.bind('<Escape>', exitKey)

    tk_root.mainloop()