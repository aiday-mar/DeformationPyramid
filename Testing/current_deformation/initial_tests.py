import copy
import re
import numpy as np
import matplotlib.pyplot as plt

# initial tests using FCGF, KNN matching and Deformation Algorithm From Astrivis

# training_data = 'full_deformed'
training_data = 'partial_deformed'

# epoch='10'
epoch='5'

data_types = ['Full Non Deformed', 'Full Deformed', 'Partial Non Deformed', 'Partial Deformed']

model_numbers = ['002', '042', '085', '126', '167', '207']

sub_matrix = {model_number : 0.0 for model_number in model_numbers}
final_data = {data_type : copy.deepcopy(sub_matrix) for data_type in data_types}

bar = np.array([0, 1, 2, 3, 4, 5])

for data_type in data_types:
    data_type_mod = data_type.lower().replace(' ', '_')
    file_name = 'Testing/current_deformation/test_astrivis_' + data_type_mod + '_current_deformation_pre_none_fcgf_td_' + training_data + '_e_' + epoch + '_knn_True_conf_1e-06.txt'
    print(file_name)
    file = open(file_name, 'r')
    lines = file.readlines()
    current_model_number = None 

    for line in lines:
        if 'model' in line and len(line) < 35:
            words = line.split(' ')
            current_model_number = words[1]
            if data_type == 'Partial Non Deformed' or data_type == 'Full Non Deformed':
                current_model_number = current_model_number[:len(current_model_number)-1]
        
        if 'RMSE' in line and current_model_number:
            list_res = re.findall("\d+\.\d+", line)
            rmse = list_res[0]
            final_data[data_type][current_model_number] = float(rmse)

    rmse_arr = []
    for model_number in model_numbers:
        rmse_arr.append(final_data[data_type][model_number])

    plt.clf()
    plt.bar(bar, rmse_arr, width = 0.9)
    plt.xticks(np.array([0, 1, 2, 3, 4, 5]), model_numbers)
    plt.xlabel('Model numbers')
    plt.ylabel('RMSE')
    plt.title(data_type)
    plt.savefig('Testing/current_deformation/initial_test_' + data_type_mod + '_fcgf_td_' + training_data + '.png')