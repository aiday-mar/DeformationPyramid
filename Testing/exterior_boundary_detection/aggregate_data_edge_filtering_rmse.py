import re
import matplotlib.pyplot as plt
import numpy as np
import copy

# FCGF TODO
# KPFCN TODO

# criterion='simple'
criterion='angle'
# criterion='shape'
# criterion='disc'

feature_extractor='kpfcn'
# feature_extractor='fcgf'

# preprocessing=none
preprocessing='mutual'

data_types=['Partial Deformed', 'Partial Non Deformed']
base = 'Testing/'
folder = 'exterior_boundary_detection/'
file='testing_' + criterion + '_edge_filtering_pre_' + preprocessing + '_' + feature_extractor + '.txt'

edge_filtering_list=['Edge filtering not used', 'Edge filtering used']
models=['002', '042', '085', '126', '167', '207']

shape = (len(edge_filtering_list),)

sub_matrix={
    'Partial Deformed': {'rmse' : np.zeros(shape), 'strict-ir' : np.zeros(shape), 'relaxed-ir' : np.zeros(shape), 'vis-epe' : np.zeros(shape), 'vis-outlier' :  np.zeros(shape)},  
    'Partial Non Deformed': {'rmse' : np.zeros(shape), 'strict-ir' : np.zeros(shape), 'relaxed-ir' : np.zeros(shape), 'vis-epe' : np.zeros(shape), 'vis-outlier' :  np.zeros(shape)}
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

    if 'Strict IR' in line and current_model is not None and edge_filtering_val is not None:
        strict_ir = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = edge_filtering_list.index(edge_filtering_val)
        final_matrices[current_model][current_data_type]['strict-ir'][i] = strict_ir

    if 'Relaxed IR' in line and current_model is not None and edge_filtering_val is not None:
        relaxed_ir = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = edge_filtering_list.index(edge_filtering_val)
        final_matrices[current_model][current_data_type]['relaxed-ir'][i] = relaxed_ir
        
    if 'vis-epe' in line and current_model is not None and edge_filtering_val is not None:
        vis_epe = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = edge_filtering_list.index(edge_filtering_val)
        final_matrices[current_model][current_data_type]['vis-epe'][i] = vis_epe
        
    if 'vis-outlier' in line and current_model is not None and edge_filtering_val is not None:
        vis_outlier = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = edge_filtering_list.index(edge_filtering_val)
        final_matrices[current_model][current_data_type]['vis-outlier'][i] = vis_outlier
        
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
    plt.xlabel('model')
    plt.ylabel('RMSE')
    plt.legend(['No Edge Filtering', 'Edge Filtering'])
    if feature_extractor == 'fcgf':
        plt.title(data_type + ' - edge filtering comparison - ' + 'FCGF feature extraction')
    elif feature_extractor == 'kpfcn':
        plt.title(data_type + ' - edge filtering comparison - ' + 'KPFCN feature extraction')
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_graph_' + feature_extractor + '_' + criterion + '.png', bbox_inches='tight')
