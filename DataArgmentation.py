import pandas as pd
import sys
from argparse import ArgumentParser

# Argument parsing setup
parser = ArgumentParser(description='Making / split training data')
parser.add_argument('-i','--inputfile', metavar='', type=str, help='Input file name', default='')
parser.add_argument('-w','--winsize', metavar='', type=int, help='Window size', default=5)
args = parser.parse_args()

if __name__ == '__main__':
    
    INPUT_FILE = args.inputfile
    WIN_SIZE = args.winsize

    if not INPUT_FILE:
        print('ERROR: You have to specify input file name!')
        sys.exit(0)
        
    target_file = INPUT_FILE
    win_size = WIN_SIZE

    # Read input file
    data = pd.read_csv(target_file, header=None, delimiter=' ')

    # Split data and label
    X = data.iloc[:, 1:801]
    Y = data.iloc[:, 0:1]

    binary_tr_filename = 'BIN_' + target_file
    steps_tr_filename = 'STEPS_' + target_file
    avg_tr_filename = 'AVGWIN_' + str(win_size) + '_' + target_file

    with open(avg_tr_filename, 'w') as avg_outfile, open(binary_tr_filename, 'w') as bin_outfile, open(steps_tr_filename, 'w') as steps_outfile:
        for index, row in X.iterrows():
            print('Processing: ' + str(index))
            # Get label
            label = Y.iloc[index].values[0]
            # Save steps only first if label is not 0
            if label > 0:
                # Write label first at the beginning of a row
                steps_outfile.write(str(label) + ' ')
                for eachvalue in row: 
                    steps_outfile.write(str('{:1.2f}'.format(eachvalue) + ' '))
                steps_outfile.write('\n')
            
            # Next, write to binary training file
            if label == 0:
                # Write label first at the beginning of a row
                bin_outfile.write('0 ')
            else:
                # Write label first at the beginning of a row
                bin_outfile.write('1 ')
            for eachvalue in row:
                bin_outfile.write(str('{:1.2f}'.format(eachvalue) + ' '))
            bin_outfile.write('\n')
            
            # Next, write to average training file
            # Get average on window
            avg_row = row.rolling(window=win_size).mean()
            # Get first value
            first_value = avg_row[win_size]
            # Fill NaN values
            avg_row[0:(win_size - 1)] = first_value
            # Write label
            avg_outfile.write(str(Y.iloc[index].values[0]) + ' ')
            # Write out avg data
            for eachvalue in avg_row:
                avg_outfile.write(str('{:1.3f}'.format(eachvalue) + ' '))
            avg_outfile.write('\n')
