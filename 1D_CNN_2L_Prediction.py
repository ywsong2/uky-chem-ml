# %%
# Import library
import pandas as pd
import numpy as np
import glob
import os
import time
import calendar
import datetime
import sys
import matplotlib.pyplot as plt
import gc

# Import Keras & machine learning libraries
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout, Activation, LSTM
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.utils import multi_gpu_model
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import h5py
from argparse import ArgumentParser

#%%
# Argument parsing setup
parser = ArgumentParser(description='Chem Project ML (1D CNN) - Two Layered Prediction')
parser.add_argument('-e','--numepoch', metavar='', type=int, help='Number of EPOCH', default=30)
parser.add_argument('-b','--batchsize', metavar='', type=int, help='Batch size', default=100)
parser.add_argument('-d','--data', metavar='', type=str, help='Data filename', default='MyLabelData.txt')
parser.add_argument('-l','--learnrate', metavar='', type=float, help='Learning rate', default=0.0001)
parser.add_argument('-p', action='store_true', help='If -p is set, it is prediction mode')
parser.add_argument('-r', action='store_true', help='If -r is set, it is using LSTM')
parser.add_argument('-a', action='store_true', help='If -a is set with -p, it will predict all files in the folder')
parser.add_argument('-n', action='store_true', help='If -n is set, it skips making images for windows sizes')
parser.add_argument('-mb','--binarymodel', metavar='', type=str, help='Model name to load for binary classification prediction', default='/home/ywsong2/Chem_ML_GUI/BINARY_CLASSIFIER_1DCNN_20190611_202715_67660.h5')
parser.add_argument('-ms','--stepmodel', metavar='', type=str, help='Model name to load for step classification prediction', default='/home/ywsong2/Chem_ML_GUI/STEP_CLASSIFIER_1DCNN_20190521_155002_49281.h5')
parser.add_argument('-i','--input', metavar='', type=str, help='Target input file to predict', default='')
parser.add_argument('-f', '--datafolder', metavar='', type=str, help='Target folder for data files', default='')
args = parser.parse_args()

#%%

# Set plt style
plt.style.use('classic')

if __name__ == '__main__':
    # Is process all enabled?
    PROCESS_ALL_DATA = False


    # Is prediction mode enabled?
    PREDMODE = True
    
    # Is this using LSTM
    USE_LSTM = False

    # Is making images?
    SKIP_MAKING_IMGS = False

    print('\n-- Flags --\n')

    # If -p is set, then this is prediction mode
    if args.p:
        print('Prediciton Mode: ON\n')
        PREDMODE = True

        # If -a is set, then this is predicting all sub-folders in the target folder
        if args.a:
            print('Predicting all data files: ON\n')
            if not args.datafolder:
                print('ERROR: Please specify target folder that contains all data files!')
                sys.exit(0)
            PROCESS_ALL_DATA = True
        else:
            print('Predicting all data files: OFF\n')
            if not args.input:
                print('ERROR: Please specify target input file!')
                sys.exit(0)
    else:
        print('Prediciton Mode: OFF\n')

    # If -n is set, skip making images
    if args.n:
        print('Skip making images: ON\n')
        SKIP_MAKING_IMGS = True
    else:
        print('Skip making images: OFF\n')

    # If -r is set, then this is RNN (LSTM)
    if args.r:
        print('LSTM: ON\n')
        USE_LSTM = True        
    else: 
        print('LSTM: OFF\n')
    
    print('\n')
    print('-- Configuration --')
    
    # If this is prediction mode, we need the model name and target input file to predict
    if PREDMODE:
        BINARY_MODELNAME = args.binarymodel
        STEP_MODELNAME = args.stepmodel
        INPUTFILE = args.input
        TARGETFOLDER = args.datafolder
        LR = args.learnrate
        
        print('Learning rate = ', LR)
        print('Binary ML model name to load = ', BINARY_MODELNAME)
        print('Step ML model name to load = ', STEP_MODELNAME)
        print('Input file name to predict = ', INPUTFILE)
        print('Target folder name to predict = ', TARGETFOLDER)
    else:
        BATCH_SIZE = args.batchsize
        NUM_EPOCH = args.numepoch
        DATA = args.data
        LR = args.learnrate

        print('Batch size = ', BATCH_SIZE)
        print('Num. of EPOCH = ', NUM_EPOCH)
        print('Learning rate = ', LR)
        print('Data = ', DATA)

    # Print out Tensorflow / Keras versions
    print('TF version = ', tf.__version__)
    print('Keras version = ', tf.keras.__version__)
    print('\n')

    loaded_step_model = False
    fig = plt.figure(figsize=(6,4))
    
    if PREDMODE:
        if len(BINARY_MODELNAME) == 0:
            print('ERROR: Specify valid target binary model name and input file name!')
            sys.exit(0)
        else:
            print('\n==========================================================')
            print(' Starting Binary Classification... (0 or Non-Zero)')
            print('==========================================================\n')
        
            print('-- Load binary model --')
            binary_model = load_model(BINARY_MODELNAME)
            binary_model.summary()
            binary_model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_v2.Adam(learning_rate=LR), metrics=['accuracy'])
            #sys.exit(0)
        
        input_files = []
                
        if PROCESS_ALL_DATA:
            for each_data_file in glob.iglob(TARGETFOLDER+'/*.xlsx'):
                input_files.append(each_data_file)
        else:
            input_files.append(INPUTFILE)

        # Debug
        print('-- List of input files --\n')
        print(input_files)

        for each_input_file in input_files:
            print('Opening up filename:', each_input_file)
            #dfs = pd.ExcelFile(each_input_file)
            dfs = pd.ExcelFile(each_input_file, engine="openpyxl")
            df = dfs.parse('Sheet1')
            df.pop('Frame Number')
        
            numberOfRows = df.shape[0]
            numberOfCols = df.shape[1]
            
            # Print out debug messages
            print('Processing:', each_input_file)
            print('Number of rows =', numberOfRows)
            print('Number of cols =', numberOfCols)
        
            # Remove if there are more than 800 rows
            if numberOfRows > 800:
                df = df.drop(df.index[800:])
            
            # Check validness of row
            if numberOfRows < 800:
                print('ERROR: Number of rows are less than 800! Invalid data set')
                continue;
                # sys.exit(0)
                
            # Check validness of column
            if numberOfCols <= 0:
                print('ERROR: Invalid number of columns in data!')
                continue;
                #sys.exit(0)
                
            # Open temporary text file to save pre-processed data
            temp_text_file = open('temp_input_binary.txt', 'w')
            temp_minmax_file = open('temp_minmax.txt', 'w')
                
            # Get each column and create temp file (each time series data)
            for eachColumn in df.columns:
                if eachColumn == 'Frame Number':
                    continue
                    
                # Get each data set, max, min values
                eachData = df[eachColumn]
                maxValue = eachData.loc[eachData.idxmax()]
                minValue = eachData.loc[eachData.idxmin()]
                # print("max:", maxValue, "min:", minValue)
                
                # Normalize data per data sample
                eachData = (eachData - minValue) / (maxValue - minValue)
                        
                # Save it to temporary text file (re-formatting)
                for i, eachValue in enumerate(eachData):
                    temp_text_file.write(format(eachValue, '.2f'))
                    if i < (len(eachData) - 1):
                        temp_text_file.write(' ')
                temp_text_file.write('\n')

                # Save min and max value for each data
                temp_minmax_file.write(format(minValue, '.2f') + ' ' + format(maxValue, '.2f') + '\n')
            
            # Make sure close temp text output file
            temp_text_file.close()

            # Close min max file
            temp_minmax_file.close()

            # Pre-processing is done
            print('Pre-processing is done')
            
            # Read in created temporary data
            input_df = pd.read_csv('temp_input_binary.txt', header=None, delimiter=' ')
            input_X = input_df.iloc[:, 0:800]
            input_X = input_X.values
            input_X = input_X.reshape(input_X.shape[0], input_X.shape[1], 1)

            # Read min max value file
            input_minmax_df = pd.read_csv('temp_minmax.txt', header=None, delimiter=' ')
            
            # Predict on test data
            classes = binary_model.predict_classes(input_X, batch_size=100)

            # Destination directory            
            dest_dir = os.path.splitext(each_input_file)[0] + '/'

            # Create folders
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                os.makedirs(dest_dir + '0/')                # 0 step
                os.makedirs(dest_dir + '1/')                # 1 step
                os.makedirs(dest_dir + '2/')                # 2 step
                os.makedirs(dest_dir + '3/')                # 3 step
                os.makedirs(dest_dir + 'NZ/')               # Not-Zero step
                os.makedirs(dest_dir + 'MultiWinSize/1')    # Multiple window size 1
                os.makedirs(dest_dir + 'MultiWinSize/2')    # Multiple window size 2
                os.makedirs(dest_dir + 'MultiWinSize/3')    # Multiple window size 3
                os.makedirs(dest_dir + 'MultiWinSize/4')    # Multiple window size 4
                os.makedirs(dest_dir + 'MultiWinSize/5')    # Multiple window size 5
            
            # Set default window size 
            default_win_size = 3

            # Data file for binary classifiaction label and also raw data
            data_file_0 = dest_dir + 'Data_0.txt'
            data_file_NZ = dest_dir + 'Data_NZ.txt'
            data_file_raw = dest_dir + 'Data.txt'

            # Open data file and write raw, 0, NZ data separately
            # Also, create all images for different window sizes (for researcher)
            with open(data_file_0, 'w') as outfile_0, open(data_file_NZ, 'w') as outfile_NZ, open(data_file_raw, 'w') as outfile_raw:
                # Loop through all the rows 
                for index, row in input_df.iterrows():
                    # Update status
                    print('Processing ', index, 'Binary Classification Result =', classes[index])
                    # print('index:', index, ' min:', input_minmax_df[0][index], ' max:', input_minmax_df[1][index])

                    # Write index
                    outfile_raw.write(str(index) + ' ')

                    # Write out data
                    for eachvalue in row:
                        outfile_raw.write(str(eachvalue) + ' ')
                    outfile_raw.write('\n')

                    # If this is 0 steps
                    if classes[index] == 0:
                        # Write index
                        outfile_0.write(str(index) + ' ')

                        # Write out data
                        for eachvalue in row:
                            outfile_0.write(str(eachvalue) + ' ')
                        outfile_0.write('\n')

                        # Set image file name 
                        dest_filename = dest_dir + '0/' + str(index) + '.png'
                    # If this is non-zero steps
                    else:
                        # Write index
                        outfile_NZ.write(str(index) + ' ') 

                        # Write out data
                        for eachvalue in row:
                            outfile_NZ.write(str(eachvalue) + ' ')
                        outfile_NZ.write('\n')
                        
                        # Set image file name 
                        dest_filename = dest_dir + 'NZ/' + str(index) + '.png'

                    if not SKIP_MAKING_IMGS:
                        # Save image to 0 or NZ folder first
                        avgData = row.rolling(window=default_win_size).mean()
                        plt.plot(avgData, color="black")
                        plt.margins(x=0.05)
                        plt.xlabel('Max:' + str(input_minmax_df[1][index]) + ' Min:' + str(input_minmax_df[0][index]))
                        fig.savefig(dest_filename, facecolor='white')
                        plt.clf()

                        # Create all PNG files for different window sizes
                        # Number of window sizes = 5 (1, 2, 3, 4, 5)
                        for win_size in range(1, 6):
                            # Save different window size based images (for dubugging)
                            dest_filename = dest_dir + 'MultiWinSize/' + str(win_size) + '/' + str(index) + '.png'
                            avgData = row.rolling(window=win_size).mean()
                            plt.plot(avgData, color="black")
                            plt.margins(x=0.05)
                            plt.ylabel('Max:' + str(input_minmax_df[1][index]) + ' Min:' + str(input_minmax_df[0][index]))
                            fig.savefig(dest_filename, facecolor='white')
                            plt.clf()
                    gc.collect()

            # Binary classification is done. Now we need to predict only NZ data set (to classify 1, 2, or 3 steps)
            print('\n==========================================================')
            print(' Binary Classification is completed! Predict Non-Zero.')
            print('==========================================================\n')

            if loaded_step_model == False:
                # Load model for step classification
                print('-- Load step classification ML model --')
                # STEP_MODELNAME = '/home/ywsong2/Chem_ML_GUI/STEP_CLASSIFIER_1DCNN_20190521_155002_49281.h5'
                step_model = load_model(STEP_MODELNAME)
                step_model.summary()
                step_model.compile(loss='categorical_crossentropy', optimizer=adam_v2.Adam(learning_rate=LR), metrics=['accuracy'])
                loaded_step_model = True
            else:
                print('WARNING: Step model was loaded before. Skiping...')

            # Read in non-zero data (only non-zero data set)
            try:
                input_df = pd.read_csv(data_file_NZ, header=None, delimiter=' ')

            except pd.errors.EmptyDataError:
                print('Note: csv was empty. Skipping.')
                continue                
                
            # Get data
            input_X = input_df.iloc[:, 1:801]
            # Get index
            index_X = input_df.iloc[:, 0:1] 

            input_X_for_ML = input_X.values
            input_X_for_ML = input_X_for_ML.reshape(input_X_for_ML.shape[0], input_X_for_ML.shape[1], 1)
            
            # Predict on test data
            classes = step_model.predict_classes(input_X_for_ML, batch_size=100)

            # Loop through all the rows (non-zero data set)
            for index, row in input_X.iterrows():
                # Update status
                print('Processing ', index, 'Step Classification Result =', classes[index]+1)

                # Get image index
                img_index = index_X.iloc[index].values[0]

                # Classified as 1 steps
                if classes[index] == 0:
                    dest_filename = dest_dir + '1/' + str(img_index) + '.png'
                # Classified as 2 steps
                elif classes[index] == 1:
                    dest_filename = dest_dir + '2/' + str(img_index) + '.png'
                # Classified as 3 steps
                else:
                    dest_filename = dest_dir + '3/' + str(img_index) + '.png'

                # Save image to each folder 
                avgData = row.rolling(window=default_win_size).mean()
                # fig = plt.figure(figsize=(6,4))
                plt.plot(avgData, color="black")
                plt.margins(x=0.05)
                fig.savefig(dest_filename, facecolor='white')
                plt.clf()

            print('\n==========================================================')
            print(' Step prediction is completed!')
            print('==========================================================\n')
        
#%%
