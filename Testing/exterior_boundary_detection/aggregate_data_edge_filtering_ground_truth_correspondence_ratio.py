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
preprocessing='none'
# preprocessing='mutual'

# current_deformation=False
current_deformation=True

bar = np.array([0, 1, 2, 3, 4, 5])

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
          
            if 'RMSE' in line and current_data_type:
                rmse = float(re.findall("\d+\.\d+", line)[0])
                final_matrices[criterion][model_number][current_data_type]['rmse'] = rmse

print(final_matrices)

for data_type in data_types:
    for model_number in model_numbers:

        plt.clf()

        true = []
        total = []

        for criterion in criteria:
            true.append(final_matrices[criterion][model_number][data_type]['true'])
            total.append(final_matrices[criterion][model_number][data_type]['total'])

        plt.bar(bar, true, color='r')
        plt.bar(bar, total, bottom=true, color='b')

        plt.title('Model ' + model_number + ' - ' + data_type)
        plt.xticks(bar, criteria)
        data_type_mod = data_type.lower()
        data_type_mod = data_type_mod.replace(' ', '_')
        if current_deformation  is False:
            filename = 'Testing/exterior_boundary_detection/ground_truth_correspondence_ratio_pre_' + preprocessing + '_model_' + model_number + '_' + data_type_mod + '_' + feature_extractor + '_td_' + training_data + '.png'
        else:
            filename = 'Testing/exterior_boundary_detection/ground_truth_correspondence_ratio_pre_' + preprocessing + '_model_' + model_number + '_' + data_type_mod + '_' + feature_extractor + '_td_' + training_data + '_current_deformation.png'

        plt.savefig()