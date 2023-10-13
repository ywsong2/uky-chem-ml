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
import seaborn as sns

# Import Keras & machine learning libraries
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout, Activation, LSTM
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.utils import multi_gpu_model
# from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import h5py
from argparse import ArgumentParser

# Example prediction commands
# python 1D_CNN_Train.py -i test.xlsx -m 20190308_092346_1DCNN_numofepoch\(20\)_batchsize\(100\)_lr\(0.0001\).h5 -p

# Example training commands
# python 1D_CNN_Train.py -e 20 -d MyLabelData.txt

# Argument parsing setup
parser = ArgumentParser(description='Chem Project ML (1D CNN) - Binary classifier (0 or not)')
parser.add_argument('-e','--numepoch', metavar='', type=int, help='Number of EPOCH', default=30)
parser.add_argument('-b','--batchsize', metavar='', type=int, help='Batch size', default=100)
parser.add_argument('-d','--data', metavar='', type=str, help='Data filename', default='MyLabelData.txt')
parser.add_argument('-l','--learnrate', metavar='', type=float, help='Learning rate', default=0.0001)
parser.add_argument('-p', action='store_true')
parser.add_argument('-r', action='store_true')
parser.add_argument('-a', action='store_true')
parser.add_argument('-m','--model', metavar='', type=str, help='Model name to load for prediction', default='')
parser.add_argument('-i','--input', metavar='', type=str, help='Target input file to predict', default='')
parser.add_argument('-f', '--datafolder', metavar='', type=str, help='Target folder for data files', default='')
args = parser.parse_args()

if __name__ == '__main__':
    
    # Is process all enabled?
    PROCESS_ALL_DATA = False

    # Is prediction mode enabled?
    PREDMODE = False
    
    # Is this using LSTM
    USE_LSTM = False

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
        MODELNAME = args.model
        INPUTFILE = args.input
        TARGETFOLDER = args.datafolder
        LR = args.learnrate
        
        print('Learning rate = ', LR)
        print('Model name to load = ', MODELNAME)
        print('Input file name to predict = ', INPUTFILE)
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
        
    if PREDMODE:
        if len(MODELNAME) == 0:
            print('ERROR: Specify valid target model name and input file name!')
            sys.exit(0)
        else:
            print('-- Load model --')
            model = load_model(MODELNAME)
            model.summary()
            model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_v2(lr=LR), metrics=['accuracy'])
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
                
            dfs = pd.ExcelFile(each_input_file)
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
                sys.exit(0)
                
            # Checi validness of column
            if numberOfCols <= 0:
                print('ERROR: Invalid number of columns in data!')
                sys.exit(0)
                
            # Open temporary text file to save pre-processed data
            temp_text_file = open('temp_input.txt', 'w')
                
            # Get each column and create temp file (each time series data)
            for eachColumn in df.columns:
                if eachColumn == 'Frame Number':
                    continue
                    
                # Get each data set, max, min values
                eachData = df[eachColumn]
                maxValue = eachData.loc[eachData.idxmax()]
                minValue = eachData.loc[eachData.idxmin()]
                
                # Normalize data per data sample
                eachData = (eachData - minValue) / (maxValue - minValue)
                        
                # Save it to temporary text file (re-formatting)
                for i, eachValue in enumerate(eachData):
                    temp_text_file.write(format(eachValue, '.2f'))
                    if i < (len(eachData) - 1):
                        temp_text_file.write(' ')
                temp_text_file.write('\n')
            
            # Make sure close temp text output file
            temp_text_file.close()

            # Pre-processing is done
            print('Pre-processing is done')
            
            # Read in created temporary data
            input_df = pd.read_csv('temp_input.txt', header=None, delimiter=' ')
            input_X = input_df.iloc[:, 0:800]
            input_X = input_X.values
            input_X = input_X.reshape(input_X.shape[0], input_X.shape[1], 1)
            
            # Predict on test data
            classes = model.predict_classes(input_X, batch_size=100)

            # Destination directory            
            dest_dir = os.path.splitext(each_input_file)[0] + '/'

            # Create folders
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                os.makedirs(dest_dir + '0/')
                os.makedirs(dest_dir + '1/')
                os.makedirs(dest_dir + '2/')
                os.makedirs(dest_dir + '3/')
                os.makedirs(dest_dir + 'MultiWinSize/1')
                os.makedirs(dest_dir + 'MultiWinSize/2')
                os.makedirs(dest_dir + 'MultiWinSize/3')
                os.makedirs(dest_dir + 'MultiWinSize/4')
                os.makedirs(dest_dir + 'MultiWinSize/5')
            
            # Set default window size 
            default_win_size = 3

            # Data file
            data_file = dest_dir + 'Data.txt'

            # Open data file
            with open(data_file, 'w') as outfile:
                # Loop through all the rows 
                for index, row in input_df.iterrows():
                    # Update status
                    print('Processing ', index)
                    
                    # Write index
                    outfile.write(str(index) + ' ')
                    
                    # Write out data
                    for eachvalue in row:
                        outfile.write(str(eachvalue) + ' ')
                    outfile.write('\n')
                    
                    # Number of window sizes = 5 (1, 2, 3, 4, 5)
                    for win_size in range(1, 6):
                        # Save different window size based images (for dubugging)
                        dest_filename = dest_dir + 'MultiWinSize/' + str(win_size) + '/' + str(index) + '.png'
                        avgData = row.rolling(window=win_size).mean()
                        plt.figure(figsize=(6,4))
                        plt.plot(avgData, color="black")
                        plt.savefig(dest_filename)

                        # If this is default window size, save it to each step folder as well
                        if win_size == default_win_size:
                            dest_filename = dest_dir + str(classes[index]) + '/' + str(index) + '.png'
                            plt.savefig(dest_filename)            
                        plt.close()
    else:
        # Read data 
        data = pd.read_csv(DATA, header=None, delimiter=' ')

        X = data.iloc[:, 1:801]
        Y = data.iloc[:, 0:1]
        #X.shape, Y.shape

        # Convert to numpy array
        X = X.values

        #train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.75, random_state=1, stratify=Y)
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.75, shuffle=True)
        # Keey Y label intact for confusion matrix calculation later
        test_Y_label = test_Y

        if USE_LSTM: 
            # Reshape into correct dimensions to input into LSTM
            # train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
            # test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

            train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1]).astype('float32')
            test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1]).astype('float32')

            train_Y = np_utils.to_categorical(train_Y, 4)
            test_Y = np_utils.to_categorical(test_Y, 4)
            
            print(train_X.shape, test_X.shape)
            # Build network layers
            model = Sequential()
            model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(1,800)))
            model.add(Dropout(0.3))
            model.add(LSTM(64, return_sequences=True, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(4, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=adam_v2(lr=LR, decay=1e-5), metrics=['accuracy'])          
        else:
            # Reshape into correct dimensions to input into CNN
            train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1).astype('float32')
            test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1).astype('float32')

            train_Y = np_utils.to_categorical(train_Y, 2)
            test_Y = np_utils.to_categorical(test_Y, 2)
            print(train_X.shape, test_X.shape)
            
            model = Sequential()
            model.add(Conv1D(32, 5, activation='relu', input_shape=(800,1)))
            model.add(Conv1D(32, 5, activation='relu'))
            model.add(Dropout(0.25))
            model.add(MaxPool1D(2))
            model.add(Conv1D(64, 10, activation='relu'))
            model.add(Conv1D(64, 10, activation='relu'))
            model.add(Dropout(0.25))
            model.add(MaxPool1D(2))
            model.add(Conv1D(128, 15, activation='relu'))
            model.add(Conv1D(128, 15, activation='relu'))
            model.add(Dropout(0.25))
            model.add(MaxPool1D(2))
            model.add(Flatten())
            model.add(Dense(2, activation='softmax'))
            print(model.summary())
            
            model.compile(loss='categorical_crossentropy', optimizer=adam_v2(lr=LR, decay=1e-5), metrics=['accuracy'])

            print(model.summary())
        
        model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=NUM_EPOCH, batch_size=BATCH_SIZE, callbacks=[])

        # score model and log accuracy and parameters
        scores = model.evaluate(test_X, test_Y, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

        # Save model
        training_mode = '1DCNN_'
        if USE_LSTM:
            training_mode = 'LSTM_'
        cur_time = time.strftime("%Y%m%d_%H%M%S")
        model_name = 'BINARY_CLASSIFIER_' + training_mode + cur_time + '_' + str(data.shape[0]) + '.h5'
        print('-- Saving model --')
        print('Model Name:', model_name)
        model.save(model_name)
        print('-- Model saved --\n')        
        
        print('--Calculate confusion matrix--')

        # Predict test data set
        test_pred_y = model.predict_classes(test_X) 

        test_Y_label = np.asarray(test_Y_label)
        test_Y_label_converted = []
        for i in np.asarray(test_Y_label):
            test_Y_label_converted.append(i[0])

        # print("test_Y:",test_Y_label_converted)
        # print("test_pred_y:",test_pred_y)

        # Create confusion matrix
        conf_matrix = confusion_matrix(test_Y_label_converted, test_pred_y)

        print(conf_matrix)
        # conf_matrix = tf.math.confusion_matrix(labels=test_Y_label_converted, predictions=test_pred_y).numpy()

        conf_matrix_norm = np.around(conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

        classes = [0,1]
        conf_matrix_df = pd.DataFrame(conf_matrix_norm, index = classes, columns = classes)

        figure = plt.figure(figsize=(8, 8))
        sns.heatmap(conf_matrix_df, annot=True,cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        conf_matrix_img_filename = 'BINARY_CLASSIFIER_' + training_mode + cur_time + '_' + str(data.shape[0]) + '_conf_matrix.png'
        plt.savefig(conf_matrix_img_filename)
        print("saved image as:",conf_matrix_img_filename)
        # plt.show()

