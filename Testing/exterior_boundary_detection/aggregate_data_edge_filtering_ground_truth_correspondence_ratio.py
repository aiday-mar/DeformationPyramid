import matplotlib.pyplot as plt
import copy

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
'''
if 'number of true landmark correspondences returned from custom filtering' in line:
    search = list(map(int, re.findall(r'\d+', line)))
    true = int(search[0])
    total = int(search[1])
    if adm_changed is True:
        final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['true'][0][count][0] = true
        final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['total'][0][count][0] = total - true
    else:
        final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['true'][count][0][0] = true
        final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['total'][count][0][0] = total - true

if 'RMSE' in line:
    rmse = float(re.findall("\d+\.\d+", line)[0])
    if adm_changed is True:
        final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['rmse'][0][count][0] = rmse
    else:
        final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['rmse'][count][0][0] = rmse
    
if 'number of distinct source landmarks ' in line:
    search = list(map(int, re.findall(r'\d+', line)))
    n_distinct = search[0]
    final_matrices[feature_extractor][training_data][model_number][current_data_type]['n_distinct'] = n_distinct
'''

for model_number in model_numbers:

    plt.clf()
    plt.savefig('Testing/exterior_boundary_detection/')