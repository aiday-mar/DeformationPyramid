import re
import matplotlib.pyplot as plt
import numpy as np
import copy

# FCGF DONE
# KPFCN TODO

feature_extractor = 'fcgf'
# feature_extractor = 'kpfcn'

models=['002', '042', '085', '126', '167', '207']
preprocessing='none'

if feature_extractor == 'fcgf':
    confidence_thresholds = [5.0e-07, 7.5e-07, 1.0e-06, 2.5e-06]
    shape = (len(confidence_thresholds),)
    sub_matrix={'Full Non Deformed': {'lepard_fcgf' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}, 
                'Full Deformed': {'lepard_fcgf' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}, 
                'Partial Deformed': {'lepard_fcgf' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}},  
                'Partial Non Deformed': {'lepard_fcgf' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}}
    type = 'lepard_fcgf'
    line_file = 'number of true landmarks correspondences returned from FCGF based Lepard'
    title = 'FCGF based Lepard'
elif feature_extractor == 'kpfcn':
    confidence_thresholds = [0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]
    shape = (len(confidence_thresholds),)
    sub_matrix={'Full Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}, 
                'Full Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}, 
                'Partial Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}},  
                'Partial Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}}
    type ='lepard'
    line_file = 'number of true landmarks correspondences returned from Lepard'
    title = 'KPFCN based Lepard'
else:
    raise Exception('Specify a valid feature extracting method')

data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
final_matrices = {model : copy.deepcopy(sub_matrix) for model in models}

sub_matrix_rmse={'Full Non Deformed': np.zeros((len(confidence_thresholds),)), 
            'Full Deformed': np.zeros((len(confidence_thresholds),)), 
            'Partial Deformed': np.zeros((len(confidence_thresholds),)),  
            'Partial Non Deformed': np.zeros((len(confidence_thresholds),))}
final_matrices_rmse = {model : copy.deepcopy(sub_matrix_rmse) for model in models}

base = 'Testing/'
file='confidence_threshold/testing_confidence_thresholds_pre_' + preprocessing + '_' + feature_extractor + '.txt'
file_txt = open(base + file, 'r')
Lines = file_txt.readlines()
confidence_threshold = -1
current_data_type = ''
current_model = ''

for line in Lines:
    if 'model ' in line:
        current_model = re.findall(r'\b\d+\b',line)[0]
        if current_model not in models:
            current_model = None
    if line[:-1] in data_types:
        current_data_type = line[:-1]
    if 'Test - confidence threshold' in line:
        confidence_threshold = float(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', line)[0])
    if line_file in line and current_model is not None:
        search = list(map(int, re.findall(r'\d+', line)))
        true = int(search[0])
        total = int(search[1])
        i = confidence_thresholds.index(confidence_threshold)
        final_matrices[current_model][current_data_type][type]['true'][i] = true
        final_matrices[current_model][current_data_type][type]['total'][i] = total - true
    if 'RMSE' in line and current_model is not None:
        list_res = re.findall("\d+\.\d+", line)
        res = list_res[0]
        i = confidence_thresholds.index(confidence_threshold)
        final_matrices_rmse[current_model][current_data_type][i] = res
    
print(final_matrices_rmse)

for data_type in data_types:
    plt.clf()
    plt.title(data_type, y=1.0)
    confidence_thresholds_pos = range(0, len(confidence_thresholds))
    plt.xticks(confidence_thresholds_pos, confidence_thresholds, rotation=90)
    plt.xlabel('model')
    plt.ylabel('RMSE')

    for model in models:
        rmse = []
        for i in range(len(confidence_thresholds)):
            rmse.append(final_matrices_rmse[model][data_type][i])
        plt.plot(confidence_thresholds_pos, rmse, color='r')
    plt.savefig('Testing/confidence_threshold/' + data_type.replace(' ', '_') + '_graph_' + feature_extractor + '_rmse.png', bbox_inches='tight')
    
    plt.clf()
    plt.title('Varying the confidence threshold - ' + data_type + ' -  ' + title, y=1.0)
    confidence_thresholds_pos = range(0, len(confidence_thresholds))
    plt.xticks(confidence_thresholds_pos, confidence_thresholds, rotation=90)
    plt.ylim(0, 1)
    plt.xlabel('confidence threshold')
    plt.ylabel('fraction of GT to all correspondences')
        
    for model in models:
        fraction = []
        for i in range(len(confidence_thresholds)):            
            if final_matrices[model][data_type][type]['total'][i] != 0:
                fraction.append(final_matrices[model][data_type][type]['true'][i]/(final_matrices[model][data_type][type]['total'][i]+final_matrices[model][data_type][type]['true'][i]))
            else:
                fraction.append(0)
        plt.plot(confidence_thresholds_pos, fraction, color='r')
    plt.savefig('Testing/confidence_threshold/' + data_type.replace(' ', '_') + '_graph_' + feature_extractor + '.png', bbox_inches='tight')
    
    plt.clf()
    plt.title('Varying the confidence threshold - ' + data_type + ' -  ' + title, y=1.0)
    plt.xlabel('confidence threshold')
    plt.ylabel('bar chart of GT to all correspondences')
    delta=[-0.6, -0.4, -0.2, 0, 0.2, 0.4]
    width = 0.2
    model_n = 0
    for model in models:
        true_data = []
        total_data = []
        
        for i in range(len(confidence_thresholds)):
            true_data.append(final_matrices[model][data_type][type]['true'][i])
            total_data.append(final_matrices[model][data_type][type]['total'][i])
                
        confidence_thresholds_pos = np.arange(0, len(confidence_thresholds))
        plt.bar(confidence_thresholds_pos + delta[model_n], true_data, width, color='r')
        plt.bar(confidence_thresholds_pos + delta[model_n], total_data, width, bottom=true_data, color='b')
        plt.xticks(confidence_thresholds_pos, confidence_thresholds, rotation=90)
        model_n += 1
        
    plt.legend(models)
    plt.savefig('Testing/confidence_threshold/' + data_type.replace(' ', '_') + '_bar_chart_'  + feature_extractor + '.png', bbox_inches='tight')
    
    