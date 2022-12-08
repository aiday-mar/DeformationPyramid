import re
import matplotlib.pyplot as plt
import numpy as np
import copy

# feature_extractor='kpfcn'
feature_extractor='fcgf'

data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'Testing/'
folder = 'samples/'
file='testing_samples_' + feature_extractor + '.txt'

samples_list=[50, 100, 500, 1000, 2000]
models=['002', '042', '085', '126', '167', '207']
shape = (len(samples_list),)

sub_matrix={'Full Non Deformed': {'rmse' : np.zeros(shape)}, 
                'Full Deformed': {'rmse' : np.zeros(shape)}, 
                'Partial Deformed': {'rmse' : np.zeros(shape)},  
                'Partial Non Deformed': {'rmse' : np.zeros(shape)}}
final_matrices = {model : copy.deepcopy(sub_matrix) for model in models}

file_txt = open(base + folder + file, 'r')
Lines = file_txt.readlines()
samples_val = -1
current_data_type = ''
current_model = None
for line in Lines:
    if 'model ' in line:
        current_model = re.findall(r'\b\d+\b',line)[0]
        if current_model not in models:
            current_model = None
            
    if 'Test - samples' in line:
        samples_val = int(re.findall('-?\d+', line)[0])
        
    if line[:-1] in data_types:
        current_data_type = line[:-1]
        
    if 'RMSE' in line and current_model is not None:
        rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = samples_list.index(samples_val)
        final_matrices[current_model][current_data_type]['rmse'][i] = rmse
        
print('final_matrices : ', final_matrices)

for data_type in data_types:
    plt.clf()
    for model in models:
        samples_pos = range(len(samples_list))
        plt.plot(samples_pos, final_matrices[model][data_type]['rmse'])
        plt.xticks(samples_pos, samples_list, rotation=90)
    
    if feature_extractor == 'fcgf':
        plt.title('Varying samples - RMSE - FCGF')
    elif feature_extractor == 'kpfcn':
        plt.title('Varying samples - RMSE - KPFCN')
        
    plt.ylabel('RMSE')
    plt.xlabel('samples')
    plt.legend(models, loc = "upper right")
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_graph_' + feature_extractor + '.png', bbox_inches='tight')