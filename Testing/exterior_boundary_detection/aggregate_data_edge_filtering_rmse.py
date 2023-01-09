import re
import matplotlib.pyplot as plt
import numpy as np
import copy

# FCGF TODO
# KPFCN TODO

feature_extractor='kpfcn'
# feature_extractor='fcgf'

preprocessing='none'
# preprocessing='mutual'

epoch = 'null'
training_data = 'pretrained'

models=['002', '042', '085', '126', '167', '207']
criteria = ['simple', 'angle', 'shape', 'disc', 'mesh', 'none']
data_types=['Partial Deformed', 'Partial Non Deformed']
base = 'Testing/'
folder = 'exterior_boundary_detection/'

sub_sub_matrix = {'rmse' : 0., 'strict-ir' : 0., 'relaxed-ir' : 0., 'vis-epe' : 0., 'vis-outlier' :  0.}
sub_matrix = {
    'Partial Deformed': {criterion : copy.deepcopy(sub_sub_matrix) for criterion in criteria},  
    'Partial Non Deformed': {criterion : copy.deepcopy(sub_sub_matrix) for criterion in criteria}
}
final_matrices = {model : copy.deepcopy(sub_matrix) for model in models}

for criterion in criteria:

    if criterion == 'none':
        file='testing_none_edge_filtering_pre_mutual_' + feature_extractor + '_td_' + training_data + '_epoch_' + epoch + '.txt'
    else:
        file='testing_' + criterion + '_edge_filtering_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_epoch_' + epoch + '.txt'
    
    file_txt = open(base + folder + file, 'r')
    Lines = file_txt.readlines()
    current_data_type = ''
    current_model = None
    criterion_val = None

    for line in Lines:
        if 'model ' in line:
            current_model = re.findall(r'\b\d+\b',line)[0]
                
        if 'using angle exterior boundary detection' in line:
            criterion_val = 'angle'

        if 'using disc exterior boundary detection' in line:
            criterion_val = 'disc'

        if 'using shape exterior boundary detection' in line:
            criterion_val = 'shape'
        
        if 'using simple exterior boundary detection' in line:
            criterion_val = 'simple'

        if 'using mesh exterior boundary detection' in line:
            criterion_val = 'mesh'

        if 'Edge filtering not used' in line:
            criterion_val = 'none'
        
        if line[:-1] in data_types:
            current_data_type = line[:-1]
            
        if 'RMSE' in line and current_model is not None and criterion_val is not None:
            rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices[current_model][current_data_type][criterion_val]['rmse'] = rmse

        if 'Strict IR' in line and current_model is not None and criterion_val is not None:
            strict_ir = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices[current_model][current_data_type][criterion_val]['strict-ir'] = strict_ir

        if 'Relaxed IR' in line and current_model is not None and criterion_val is not None:
            relaxed_ir = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices[current_model][current_data_type][criterion_val]['relaxed-ir'] = relaxed_ir
            
        if 'vis-epe' in line and current_model is not None and criterion_val is not None:
            vis_epe = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices[current_model][current_data_type][criterion_val]['vis-epe'] = vis_epe
            
        if 'vis-outlier' in line and current_model is not None and criterion_val is not None:
            vis_outlier = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices[current_model][current_data_type][criterion_val]['vis-outlier'] = vis_outlier
            
for data_type in data_types:
    plt.clf()
    for criterion in criteria:
        criterion_res = []

        for model_number in models:
            criterion_res.append(final_matrices[model_number][data_type][criterion]['rmse'])
        
        print(criterion_res)
        plt.plot(models, criterion_res)
    
    plt.title(data_type)
    plt.ylabel('RMSE')
    plt.xlabel('model numbers')
    plt.legend(criteria, loc = "upper right")
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_rmse_' + feature_extractor + '.png', bbox_inches='tight')
