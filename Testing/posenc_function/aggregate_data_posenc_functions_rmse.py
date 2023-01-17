import re
import matplotlib.pyplot as plt
import numpy as np
import copy

# FCGF DONE
# KPFCN TODO

feature_extractor='kpfcn'
# feature_extractor='fcgf'

barWidth = 0.13
barWidthPlot = 0.13
br1 = np.array([0, 1, 2, 3, 4])
br2 = np.array([x + barWidth for x in br1])
br3 = np.array([x + barWidth for x in br2])
br4 = np.array([x + barWidth for x in br3])
br5 = np.array([x + barWidth for x in br4])
br6 = np.array([x + barWidth for x in br5])
final_bars = [br1, br2, br3, br4, br5, br6]

training_data='pretrained'
preprocessing='mutual'
data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'Testing/'
folder = 'posenc_function/'
file='testing_posenc_functions_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '.txt'

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
        
for data_type in data_types:
    plt.clf()
    index = 0
    for model in models:
        bar = final_bars[index]
        posenc_function_pos = range(len(posenc_functions_list))
        print(bar)
        print( final_matrices[model][data_type]['rmse'])
        plt.bar(bar, final_matrices[model][data_type]['rmse'], width=barWidthPlot)
        index += 1
    
    plt.xticks([r + barWidthPlot for r in range(len(posenc_functions_list))], posenc_functions_list)
    plt.title(data_type)   
    plt.ylabel('RMSE')
    plt.xlabel('positional encoding')
    plt.legend(models, loc = "upper right")
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_graph_' + feature_extractor + '.png', bbox_inches='tight')