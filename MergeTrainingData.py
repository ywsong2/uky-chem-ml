# %%
import shutil
import time
import os

list_of_data = [
    # 'Data/MergedData_100821.txt',
    # 'Data/MergedData_102221.txt',
    # 'Data/MergedData_102521.txt',
    # 'Data/MergedData_120921.txt',
    # 'Data/MergedData_011322.txt',
    # 'Data/MergedData_021622.txt',
    # 'Data/MergedData_031122.txt',
    # 'Data/MergedData_042022.txt',
]

cur_time = time.strftime("%Y%m%d_%H%M%S")
out_filename = 'TWSD_' + cur_time + '.txt'

with open(out_filename, 'wb') as wfd:
    for f in list_of_data:
        with open(f, 'rb') as fd:
            print("Reading: ", f)
            shutil.copyfileobj(fd, wfd)

# Count number of lines (to calculate total training data size)
line_counter = 0
with open(out_filename, 'r') as of:
    num_of_lines = len(of.readlines())
    print("Num of lines in ", of.name, " = ", num_of_lines)
    line_counter = num_of_lines

out_filename_with_count = 'TWSD_' + cur_time + '_' + str(line_counter) + '.txt'
os.rename(out_filename, out_filename_with_count)