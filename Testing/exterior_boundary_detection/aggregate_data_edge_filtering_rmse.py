import matplotlib.pyplot as plt
import copy
import re
import numpy as np

criteria = ['none', 'mesh', 'shape', 'angle', 'disc', 'simple']
data_types = ['Partial Deformed', 'Partial Non Deformed']
model_numbers=['002', '042', '085', '126', '167', '207']

feature_extractor='fcgf'
# feature_extractor='kpfcn'
training_data='partial_deformed'
epoch='5'
# preprocessing='mutual'
preprocessing='none'

current_deformation = True
# current_deformation = False
barWidth = 0.10
br1 = np.array([0, 1, 2, 3, 4, 5])
br2 = np.array([x + barWidth for x in br1])
br3 = np.array([x + barWidth for x in br2])
br4 = np.array([x + barWidth for x in br3])
br5 = np.array([x + barWidth for x in br4])
br6 = np.array([x + barWidth for x in br5])
bars = [br1, br2, br3, br4, br5, br6]

final_sub_sub_sub_matrix = {'true' : 0, 'total' : 0, 'rmse' : 0.0}
final_sub_sub_matrix = {data_type : copy.deepcopy(final_sub_sub_sub_matrix) for data_type in data_types}
final_submatrix = {model_number : copy.deepcopy(final_sub_sub_matrix) for model_number in model_numbers}
final_matrices = {criterion : copy.deepcopy(final_submatrix) for criterion in criteria}

for criterion in criteria:
    for model_number in model_numbers:
        if current_deformation is False:
            file_txt = 'Testing/exterior_boundary_detection/testing_' + criterion + '_edge_filtering_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_epoch_' + epoch + '.txt'
        else:
            file_txt = 'Testing/exterior_boundary_detection/testing_' + criterion + '_edge_filtering_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_epoch_' + epoch + '_current_deformation.txt'
        
        file_txt = open(file_txt, 'r')
        Lines = file_txt.readlines()
        
        current_model = None
        current_data_type = None

        for line in Lines:
            if 'model' in line and len(line) < 100:
                current_model = re.findall(r'\b\d+\b',line)[0]
            
            for data_type in data_types:
                if data_type in line:
                    current_data_type = data_type

            if criterion == 'none':
                if feature_extractor == 'kpfcn':
                    line_to_check = 'number of true landmark correspondences returned from Outlier Rejection'
                else:
                    line_to_check = 'number of true landmark correspondences returned from FCGF based Outlier Rejection'
                
                if line_to_check in line and current_data_type and current_model:
                    search = list(map(int, re.findall(r'\d+', line)))
                    true = int(search[0])
                    total = int(search[1])
                    final_matrices[criterion][current_model][current_data_type]['true'] = true
                    final_matrices[criterion][current_model][current_data_type]['total'] = total - true
            
            else:

                line_to_check = 'number of true landmark correspondences returned from ' + criterion + ' edge filtering'
                if line_to_check in line and current_data_type and current_model:
                    search = list(map(int, re.findall(r'\d+', line)))
                    true = int(search[0])
                    total = int(search[1])
                    final_matrices[criterion][current_model][current_data_type]['true'] = true
                    final_matrices[criterion][current_model][current_data_type]['total'] = total - true
          
            if 'RMSE' in line and current_data_type and current_model:
                rmse = float(re.findall("\d+\.\d+", line)[0])
                final_matrices[criterion][current_model][current_data_type]['rmse'] = rmse

for data_type in data_types:

    plt.clf()

    count = 0
    for criterion in criteria:
    
        rmse = []
        for model_number in model_numbers:
            rmse.append(final_matrices[criterion][model_number][data_type]['rmse'])

        bar = bars[count]
        plt.bar(bar, rmse, width = barWidth, label = criterion)

        count += 1

    plt.title(data_type)
    plt.xticks([0, 1, 2, 3, 4, 5], model_numbers)
    plt.xlabel('Model number')
    plt.ylabel('RMSE')
    plt.legend(loc = 'upper right')
    data_type_mod = data_type.lower()
    data_type_mod = data_type_mod.replace(' ', '_')

    if current_deformation is False:
        filename = 'Testing/exterior_boundary_detection/rmse_pre_' + preprocessing + '_' + data_type_mod + '_' + feature_extractor + '_td_' + training_data + '.png'
    else:
        filename = 'Testing/exterior_boundary_detection/rmse_pre_' + preprocessing + '_' + data_type_mod + '_' + feature_extractor + '_td_' + training_data + '_current_deformation.png'
    
    plt.savefig(filename)
