import pandas as pd
import numpy as np
import glob
import os
import time
from argparse import ArgumentParser

parser = ArgumentParser(description='Chem Project ML (1D CNN) - Split training data into binary and step')
parser.add_argument('-d','--data', metavar='', type=str, help='Data filename', default='MyLabelData.txt')
args = parser.parse_args()

if __name__ == '__main__':
    INPUTDATA = args.data

    binary_train_data_file_name = 'BINARY_' + INPUTDATA
    step_train_data_file_name = 'STEPS_' + INPUTDATA

    print('\n')
    print('-- Configuration --')
    print('Target file:', INPUTDATA)
    print('Binary training data file name will be:', binary_train_data_file_name)
    print('Step training data file name will be:', step_train_data_file_name)
    
    # Read in input data file (all meraged data with classification labels)
    input_data = pd.read_csv(INPUTDATA, header=None, delimiter=' ')
    data = input_data.iloc[:, 1:801]
    label = input_data.iloc[:, 0:1]

    # Create training data for binary and steps (at the same time)
    with open(binary_train_data_file_name, 'w') as btf, open(step_train_data_file_name, 'w') as stf:
        for index, row in data.iterrows():
            if index % 10000 == 0:
                print('Processing:' + str(index))

            binary_label = 0
            step_label = 0
            write_step = False

            if label.iloc[index].values[0] == 0:
                binary_label = 0
            else:
                binary_label = 1
                step_label = label.iloc[index].values[0]
                write_step = True

            # Update binary label
            btf.write(str(binary_label) + ' ')

            # Write out data
            for eachvalue in row:
                btf.write(str('{:1.2f}'.format(eachvalue) + ' '))
            btf.write('\n')

            # Update step label
            if write_step:
                stf.write(str(step_label) + ' ')

                # Write out data to step training data
                for eachvalue in row:
                    stf.write(str('{:1.2f}'.format(eachvalue) + ' '))
                stf.write('\n')

