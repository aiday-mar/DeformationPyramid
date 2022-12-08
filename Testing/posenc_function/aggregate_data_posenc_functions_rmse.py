import re
import matplotlib.pyplot as plt
import numpy as np
import copy

# FCGF DONE
# KPFCN TODO

# feature_extractor='kpfcn'
feature_extractor='fcgf'

data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'Testing/'
folder = 'posenc_function/'
file='testing_posenc_functions_' + feature_extractor + '.txt'

posenc_functions_list=['log', 'linear', 'square', 'power2', 'power4']
models=['002', '042', '085', '126', '167', '207']
shape = (len(posenc_functions_list),)

sub_matrix={'Full Non Deformed': {'rmse' : np.zeros(shape)}, 
                'Full Deformed': {'rmse' : np.zeros(shape)}, 
                'Partial Deformed': {'rmse' : np.zeros(shape)},  
                'Partial Non Deformed': {'rmse' : np.zeros(shape)}}
final_matrices = {model : copy.deepcopy(sub_matrix) for model in models}

file_txt = open(base + folder + file, 'r')
Lines = file_txt.readlines()
posenc_function_val = ''
current_data_type = ''
current_model = None
for line in Lines:
    if 'model ' in line:
        current_model = re.findall(r'\b\d+\b',line)[0]
        if current_model not in models:
            current_model = None
            
    if 'Test - positional encoding' in line:
        if 'linear' in line:
            posenc_function_val = 'linear'
        elif 'none' in line:
            posenc_function_val = 'power2'
        elif 'power4' in line:
            posenc_function_val = 'power4'
        elif 'log' in line:
            posenc_function_val = 'log'
        elif 'square' in line:
            posenc_function_val = 'square'
            
    if line[:-1] in data_types:
        current_data_type = line[:-1]
        
    if 'RMSE' in line and current_model is not None:
        rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = posenc_functions_list.index(posenc_function_val)
        final_matrices[current_model][current_data_type]['rmse'][i] = rmse
        
print('final_matrices : ', final_matrices)

for data_type in data_types:
    plt.clf()
    for model in models:
        posenc_function_pos = range(len(posenc_functions_list))
        plt.plot(posenc_function_pos, final_matrices[model][data_type]['rmse'])
        plt.xticks(posenc_function_pos, posenc_functions_list, rotation=90)
    
    if feature_extractor == 'fcgf':
        plt.title('Varying positional encoding - RMSE - FCGF')
    elif feature_extractor == 'kpfcn':
        plt.title('Varying positional encoding - RMSE - KPFCN')
        
    plt.ylabel('RMSE')
    plt.xlabel('positional encoding')
    plt.legend(models, loc = "upper right")
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_graph_' + feature_extractor + '.png', bbox_inches='tight')