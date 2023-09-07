# %%
import shutil
import time
import os

list_of_data = [
    # 'Data/MergedData_surya_100821.txt',
    # 'Data/MergedData_surya_102221.txt',
    # 'Data/MergedData_surya_102521.txt',
    # 'Data/MergedData_surya_120921.txt',
    # 'Data/MergedData_Surya_011322.txt',
    # 'Data/MergedData_Surya_021622.txt',
    # 'Data/MergedData_Surya_031122.txt',
    # 'Data/MergedData_Surya_042022.txt',
    'Data/XuFu/XuFu_cerebellum_MergedData.txt',
    'Data/XuFu/XuFu_highConcentration_saline_cerebellum_MergedData.txt',
    'Data/XuFu/XuFu_highConcentration_saline_cortex_MergedData.txt',
    'Data/XuFu/XuFu_highConcentration_saline_hippocampus_MergedData.txt',
    'Data/XuFu/XuFu_highConcentration_saline_midbrain_MergedData.txt',
    'Data/XuFu/XuFu_highConcentration_saline_striatum_MergedData.txt',
    'Data/XuFu/XuFu_highConcentration_saline_thalaums_MergedData.txt',
    'Data/XuFu/XuFu_highConcentration_saline_whoeBrain_MergedData.txt',
    'Data/SyntheticData/20190510_135816_syn_data.txt',   
    # 'Data/Qianye/MergedData_Qianye_03072023_partail.txt', 
    'Data/Qianye/MergedData_Qianye_6K_2023_06_20.txt',
    'Data/Qianye/MergedData_Qianye_3K_2023_06_28.txt',

    

    # === Below is previous data ===
    # 'Data/cerebellum/MergedData.txt',
    # 'Data/high concentration/saline/cerebellum/MergedData.txt',
    # 'Data/high concentration/saline/cortex/MergedData.txt',
    # 'Data/high concentration/saline/hippocampus/MergedData.txt',
    # 'Data/high concentration/saline/midbrain/MergedData.txt',
    # 'Data/high concentration/saline/striatum/MergedData.txt',
    # 'Data/high concentration/saline/thalamus/MergedData.txt',
    # 'Data/high concentration/saline/whole brain/MergedData.txt',
    # '20190510_135816_syn_data.txt'
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