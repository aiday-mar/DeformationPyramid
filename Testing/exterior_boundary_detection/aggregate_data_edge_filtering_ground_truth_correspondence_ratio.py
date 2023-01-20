import matplotlib.pyplot as plt
import copy
import re

criteria = ['simple', 'angle', 'shape', 'disc', 'mesh', 'none']
model_numbers=['002', '042', '085', '126', '167', '207']

feature_extractor='fcgf'
training_data='full_deformed'
epoch='5'
preprocessing='mutual'

final_sub_sub_matrix = {'true' : 0, 'total' : 0, 'rmse' : 0}
final_submatrix = {model_number : copy.deepcopy(final_sub_sub_matrix) for model_number in model_numbers}
final_matrices = {criterion : copy.deepcopy(final_submatrix) for criterion in criteria}

for criterion in criteria:
    for model_number in model_numbers:
        file_txt = 'Testing/exterior_boundary_detection/testing_' + criterion + '_edge_filtering_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_epoch_' + epoch + '.txt'
        file_txt = open(file_txt, 'r')
        Lines = file_txt.readlines()

        for line in Lines:
            if 'model' in line and len(line) < 100:
                current_model = re.findall(r'\b\d+\b',line)[0]
            
            line_to_check = 'number of true landmark correspondences returned from ' + criterion + ' edge filtering'
            if line_to_check in line:
                search = list(map(int, re.findall(r'\d+', line)))
                true = int(search[0])
                total = int(search[1])
                final_matrices[criterion][model_number]['true'] = true
                final_matrices[criterion][model_number]['total'] = total - true
          
            if 'RMSE' in line:
                rmse = float(re.findall("\d+\.\d+", line)[0])
                final_matrices[criterion][model_number]['rmse'] = rmse

print(final_matrices)

for model_number in model_numbers:

    plt.clf()
    # plt.savefig('Testing/exterior_boundary_detection/')