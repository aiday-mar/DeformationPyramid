import re
import matplotlib.pyplot as plt
import numpy as np
import copy

preprocessing='none'
# preprocessing='mutual'

weights = {
    'weights1': {
        'feature_extractor': 'kpfcn',
        'training_data': 'pretrained',
        'epoch': 'null'

    },
    'weights2': {
        'feature_extractor': 'kpfcn',
        'training_data': 'partial_deformed',
        'epoch': '5'

    },
    'weights3': {
        'feature_extractor': 'fcgf',
        'training_data': 'partial_deformed',
        'epoch': '5'

    }
}

legend = ['kpfcn - pretrained', 'kpfcn - partial deformed', 'fcgf - partial deformed']

criterion = 'mesh'
models=['002', '042', '085', '126', '167', '207']
data_types=['Partial Deformed', 'Partial Non Deformed']
base = 'Testing/'
folder = 'exterior_boundary_detection/'

sub_sub_matrix = {'rmse' : 0., 'strict-ir' : 0., 'relaxed-ir' : 0., 'vis-epe' : 0., 'vis-outlier' :  0.}
sub_matrix = {
    'Partial Deformed': {weight : copy.deepcopy(sub_sub_matrix) for weight in weights},  
    'Partial Non Deformed': {weight : copy.deepcopy(sub_sub_matrix) for weight in weights}
}
final_matrices = {model : copy.deepcopy(sub_matrix) for model in models}
final_matrices_initial = {model : copy.deepcopy(sub_matrix) for model in models}

for weights_name in weights:

    weights_data = weights[weights_name]
    feature_extractor = weights_data['feature_extractor']
    training_data = weights_data['training_data']
    epoch = weights_data['epoch']

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
               
        if line[:-1] in data_types:
            current_data_type = line[:-1]
            
        if 'RMSE' in line and current_model is not None:
            rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices[current_model][current_data_type][weights_name]['rmse'] = rmse

        if 'Strict IR' in line and current_model is not None:
            strict_ir = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices[current_model][current_data_type][weights_name]['strict-ir'] = strict_ir

        if 'Relaxed IR' in line and current_model is not None:
            relaxed_ir = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices[current_model][current_data_type][weights_name]['relaxed-ir'] = relaxed_ir
            
        if 'vis-epe' in line and current_model is not None:
            vis_epe = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices[current_model][current_data_type][weights_name]['vis-epe'] = vis_epe
            
        if 'vis-outlier' in line and current_model is not None:
            vis_outlier = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices[current_model][current_data_type][weights_name]['vis-outlier'] = vis_outlier
    
    # For initial
    file='test_astrivis_partial_deformed_pre_mutual_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + '.txt'    
    file_txt = open(base + folder + file, 'r')
    Lines = file_txt.readlines()
    current_data_type = 'Partial Deformed'
    current_model = None
    criterion_val = None

    for line in Lines:
        if 'model ' in line:
            current_model = re.findall(r'\b\d+\b',line)[0]
               
        if line[:-1] in data_types:
            current_data_type = line[:-1]
            
        if 'RMSE' in line and current_model is not None:
            rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices_initial[current_model][current_data_type][weights_name]['rmse'] = rmse

        if 'Strict IR' in line and current_model is not None:
            strict_ir = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices_initial[current_model][current_data_type][weights_name]['strict-ir'] = strict_ir

        if 'Relaxed IR' in line and current_model is not None:
            relaxed_ir = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices_initial[current_model][current_data_type][weights_name]['relaxed-ir'] = relaxed_ir
            
        if 'vis-epe' in line and current_model is not None:
            vis_epe = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices_initial[current_model][current_data_type][weights_name]['vis-epe'] = vis_epe
            
        if 'vis-outlier' in line and current_model is not None:
            vis_outlier = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices_initial[current_model][current_data_type][weights_name]['vis-outlier'] = vis_outlier
        
    file='test_astrivis_partial_deformed_pre_mutual_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + '.txt'
    file_txt = open(base + folder + file, 'r')
    Lines = file_txt.readlines()
    current_data_type = 'Partial Non Deformed'
    current_model = None
    criterion_val = None

    for line in Lines:
        if 'model ' in line:
            current_model = re.findall(r'\b\d+\b',line)[0]
               
        if line[:-1] in data_types:
            current_data_type = line[:-1]
            
        if 'RMSE' in line and current_model is not None:
            rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices_initial[current_model][current_data_type][weights_name]['rmse'] = rmse

        if 'Strict IR' in line and current_model is not None:
            strict_ir = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices_initial[current_model][current_data_type][weights_name]['strict-ir'] = strict_ir

        if 'Relaxed IR' in line and current_model is not None:
            relaxed_ir = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices_initial[current_model][current_data_type][weights_name]['relaxed-ir'] = relaxed_ir
            
        if 'vis-epe' in line and current_model is not None:
            vis_epe = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices_initial[current_model][current_data_type][weights_name]['vis-epe'] = vis_epe
            
        if 'vis-outlier' in line and current_model is not None:
            vis_outlier = list(map(float, re.findall("\d+\.\d+", line)))[0]
            final_matrices_initial[current_model][current_data_type][weights_name]['vis-outlier'] = vis_outlier

colors = ['blue', 'orange', 'green']

for data_type in data_types:
    
    color_idx = 0
    plt.clf()
    for weights_name in weights:
        res = []
        for model_number in models:
            res.append(final_matrices[model_number][data_type][weights_name]['rmse'])
        plt.plot(models, res, color = colors[color_idx])

        res = []
        for model_number in models:
            res.append(final_matrices_initial[model_number][data_type][weights_name]['rmse'])
        plt.plot(models, res, color = colors[color_idx], linestyle='dashed', label='_nolegend_')
    
        color_idx += 1

    plt.title(data_type)
    plt.ylabel('RMSE')
    plt.xlabel('model numbers')
    plt.legend(legend, loc = "upper right")
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_rmse_different_weights.png', bbox_inches='tight')
