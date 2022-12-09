import re
import matplotlib.pyplot as plt
import numpy as np
import copy

# FCGF TODO
# KPFCN TODO

feature_extractor='kpfcn'
# feature_extractor='fcgf'

# preprocessing=none
preprocessing='mutual'

data_types=['Partial Deformed', 'Partial Non Deformed']
base = 'Testing/'
folder = 'edge_filtering/'
file='testing_edge_filtering_pre_' + preprocessing + '_' + feature_extractor + '.txt'

edge_filtering_list=['Edge filtering not used', 'Edge filtering used']
models=['002', '042', '085', '126', '167', '207']

shape = (len(edge_filtering_list),)

sub_matrix={
    'Partial Deformed': {'rmse' : np.zeros(shape), 'ir' : np.zeros(shape), 'vis-epe' : np.zeros(shape), 'vis-outlier' :  np.zeros(shape)},  
    'Partial Non Deformed': {'rmse' : np.zeros(shape), 'ir' : np.zeros(shape), 'vis-epe' : np.zeros(shape), 'vis-outlier' :  np.zeros(shape)}
}
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

for data_type in data_types:
    plt.clf()
    rmse_no_edge_filtering = []
    rmse_edge_filtering = []
    
    for model in models:
        rmse_no_edge_filtering.append(final_matrices[model][data_type]['rmse'][0])
        rmse_edge_filtering.append(final_matrices[model][data_type]['rmse'][1])
    
    print(rmse_no_edge_filtering)
    print(rmse_edge_filtering)
    plt.plot(models, rmse_no_edge_filtering)
    plt.plot(models, rmse_edge_filtering)
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_graph_' + feature_extractor + '.png', bbox_inches='tight')
