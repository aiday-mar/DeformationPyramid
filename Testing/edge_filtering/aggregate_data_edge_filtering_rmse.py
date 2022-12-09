import re
import matplotlib.pyplot as plt
import numpy as np
import copy

# FCGF TODO
# KPFCN TODO

# feature_extractor='kpfcn'
feature_extractor='fcgf'

# preprocessing=none
preprocessing='mutual'

data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'Testing/'
folder = 'edge_filtering/'
file='testing_edge_filtering_pre_' + preprocessing + '_' + feature_extractor + '.txt'

edge_filtering_list=['Edge filtering not used', 'Edge filtering used']
models=['002', '042', '085', '126', '167', '207']
shape = (len(edge_filtering_list),)

sub_matrix={'Full Non Deformed': {'rmse' : np.zeros(shape), 'ir' : np.zeros(shape), 'vis-epe' : np.zeros(shape), 'vis-outlier' :  np.zeros(shape)}, 
                'Full Deformed': {'rmse' : np.zeros(shape), 'ir' : np.zeros(shape), 'vis-epe' : np.zeros(shape), 'vis-outlier' :  np.zeros(shape)}, 
                'Partial Deformed': {'rmse' : np.zeros(shape), 'ir' : np.zeros(shape), 'vis-epe' : np.zeros(shape), 'vis-outlier' :  np.zeros(shape)},  
                'Partial Non Deformed': {'rmse' : np.zeros(shape), 'ir' : np.zeros(shape), 'vis-epe' : np.zeros(shape), 'vis-outlier' :  np.zeros(shape)}}
final_matrices = {model : copy.deepcopy(sub_matrix) for model in models}

file_txt = open(base + folder + file, 'r')
Lines = file_txt.readlines()
current_data_type = ''
current_model = None
edge_filtering_val = None

for line in Lines:
    if 'model ' in line:
        current_model = re.findall(r'\b\d+\b',line)[0]
        if current_model not in models:
            current_model = None
            
    if 'Edge filtering not used' in line:
        edge_filtering_val = 'Edge filtering not used'
    elif 'Edge filtering used' in line:
        edge_filtering_val = 'Edge filtering used'
              
    if line[:-1] in data_types:
        current_data_type = line[:-1]
        
    if 'RMSE' in line and current_model is not None and edge_filtering_val is not None:
        rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = edge_filtering_list.index(edge_filtering_val)
        final_matrices[current_model][current_data_type]['rmse'][i] = rmse

    if 'IR' in line and current_model is not None and edge_filtering_val is not None:
        rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = edge_filtering_list.index(edge_filtering_val)
        final_matrices[current_model][current_data_type]['ir'][i] = rmse

    if 'vis-epe' in line and current_model is not None and edge_filtering_val is not None:
        rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = edge_filtering_list.index(edge_filtering_val)
        final_matrices[current_model][current_data_type]['vis-epe'][i] = rmse
        
    if 'vis-outlier' in line and current_model is not None and edge_filtering_val is not None:
        rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = edge_filtering_list.index(edge_filtering_val)
        final_matrices[current_model][current_data_type]['vis-outlier'][i] = rmse
        
print('final_matrices : ', final_matrices)

'''
for data_type in data_types:
    plt.clf()
    for model in models:
        posenc_function_pos = range(len(edge_filtering_list))
        plt.plot(posenc_function_pos, final_matrices[model][data_type]['rmse'])
        plt.xticks(posenc_function_pos, edge_filtering_list, rotation=90)
    
    if feature_extractor == 'fcgf':
        plt.title('Varying positional encoding - ' + data_type + ' - RMSE - FCGF')
    elif feature_extractor == 'kpfcn':
        plt.title('Varying positional encoding - ' + data_type + ' - RMSE - KPFCN')
        
    plt.ylabel('RMSE')
    plt.xlabel('positional encoding')
    plt.legend(models, loc = "upper right")
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_graph_' + feature_extractor + '.png', bbox_inches='tight')
'''