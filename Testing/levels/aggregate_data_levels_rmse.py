import re
import matplotlib.pyplot as plt
import numpy as np
import copy

# feature_extractor='kpfcn'
feature_extractor='fcgf' # DONE

data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'Testing/'
folder = 'levels/'
file='testing_levels_' + feature_extractor + '.txt'

levels_list=[2, 4, 6, 8, 10]
models=['002', '042', '085', '126', '167', '207']
shape = (len(levels_list),)

sub_matrix={'Full Non Deformed': {'rmse' : np.zeros(shape)}, 
                'Full Deformed': {'rmse' : np.zeros(shape)}, 
                'Partial Deformed': {'rmse' : np.zeros(shape)},  
                'Partial Non Deformed': {'rmse' : np.zeros(shape)}}
final_matrices = {model : copy.deepcopy(sub_matrix) for model in models}

file_txt = open(base + folder + file, 'r')
Lines = file_txt.readlines()
levels_val = -1
current_data_type = ''
current_model = None
for line in Lines:
    if 'model ' in line:
        current_model = re.findall(r'\b\d+\b',line)[0]
        if current_model not in models:
            current_model = None
            
    if 'Test - levels' in line:
        levels_val = int(re.findall('-?\d+', line)[1])
        
    if line[:-1] in data_types:
        current_data_type = line[:-1]
        
    if 'RMSE' in line and current_model is not None:
        rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = levels_list.index(levels_val)
        final_matrices[current_model][current_data_type]['rmse'][i] = rmse
        
print('final_matrices : ', final_matrices)

for data_type in data_types:
    plt.clf()
    for model in models:
        k0_pos = range(len(levels_list))
        plt.plot(k0_pos, final_matrices[model][data_type]['rmse'])
        plt.xticks(k0_pos, levels_list, rotation=90)
    
    if feature_extractor == 'fcgf':
        plt.title('Varying levels - RMSE - FCGF')
    elif feature_extractor == 'kpfcn':
        plt.title('Varying levels - RMSE - KPFCN')
        
    plt.ylabel('RMSE')
    plt.xlabel('levels')
    plt.legend(models, loc = "upper right")
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_graph_' + feature_extractor + '.png', bbox_inches='tight')