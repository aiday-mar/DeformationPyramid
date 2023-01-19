import re
import matplotlib.pyplot as plt
import numpy as np
import copy

class File:
    def __init__(self, type, preprocessing, confidence, number_centers, average_distance_multiplier, coarse_level, index_coarse, number_iterations, inlier_outlier_thr, sampling):
        self.type = type
        self.preprocessing = preprocessing
        self.confidence = confidence
        self.number_centers = number_centers
        self.average_distance_multiplier = average_distance_multiplier
        self.coarse_level = coarse_level
        self.index_coarse = index_coarse
        self.number_iterations = number_iterations
        self.inlier_outlier_thr = inlier_outlier_thr
        self.sampling = sampling
    
    def __str__(self):
        return " - Type : " + str(self.type) + " - Preprocessing : " + str(self.preprocessing) + " - Confidence : " + str(self.confidence) + " - Number Centers : " + str(self.number_centers) + " - Average Distance Multiplier : " + str(self.average_distance_multiplier) + " - Coarse Level : " + str(self.coarse_level) + " - Index Coarse : " + str(self.index_coarse) + ' - Number Iterations : ' + str(self.number_iterations) + ' - Inlier Outlier Threshold : ' + str(self.inlier_outlier_thr) + ' - Sampling : ' + str(self.sampling)

files=[]
file_types=[]
number_iterations=1
data_types=['Full Deformed', 'Partial Deformed']
base = 'Testing/custom_filtering/'

type='kpfcn'

# nc = [100, 200, 300]:
nc = [10, 50, 100, 300, 500]

# adm = [1, 2, 3, 4]
# adm =  [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0]
# adm = [1.0, 2.0, 3.0, 4.0, 5.0]
adm = [3.0]

# iot=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
# iot = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
iot = [0.01]

# sampling = 'poisson'
sampling='linspace'

version=4

# adm_changed=True
adm_changed=False

# model_number='002'
model_numbers = ['002', '042']

shape=(len(nc), len(adm), len(iot))
final_submatrix ={
                'Full Deformed': {'lepard' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'outlier' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}, 'n_distinct' : 0}, 
                'Partial Deformed': {'lepard' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'outlier' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}, 'n_distinct' : 0},  
                }

final_matrices = { model_number : copy.deepcopy(final_submatrix) for model_number in model_numbers}

for model_number in model_numbers:
    count = 0
    for i in nc :
        for j in adm:
            for k in iot:
                file = 'v_' + str(version) + '_t_custom_p_none_c_0.00000001_nc_' + str(i) + '_adm_' + str(j) + '_cl_-2_ic_1_ni_' + str(number_iterations) + '_iot_' + str(k) + '_s_' + sampling + '_' + type + '_model_' + model_number + '.txt'
                files.append(file)
                file_types.append(File('custom', 'none', 0.1, i, j, -2, 1, number_iterations, k, sampling))
                file_txt = open(base + file, 'r')
                Lines = file_txt.readlines()
                current_data_type = ''
                for line in Lines:
                    if line[:-1] in data_types:
                        current_data_type = line[:-1]
                            
                    if 'number of true landmark correspondences returned from custom filtering' in line:
                        search = list(map(int, re.findall(r'\d+', line)))
                        true = int(search[0])
                        total = int(search[1])
                        if adm_changed is True:
                            final_matrices[model_number][current_data_type]['custom']['true'][0][count][0] = true
                            final_matrices[model_number][current_data_type]['custom']['total'][0][count][0] = total - true
                        else:
                            final_matrices[model_number][current_data_type]['custom']['true'][count][0][0] = true
                            final_matrices[model_number][current_data_type]['custom']['total'][count][0][0] = total - true
                    
                    if 'RMSE' in line:
                        rmse = float(re.findall("\d+\.\d+", line)[0])
                        if adm_changed is True:
                            final_matrices[model_number][current_data_type]['custom']['rmse'][0][count][0] = rmse
                        else:
                            final_matrices[model_number][current_data_type]['custom']['rmse'][count][0][0] = rmse
                        
                    if 'number of distinct source landmarks ' in line:
                        search = list(map(int, re.findall(r'\d+', line)))
                        n_distinct = search[0]
                        final_matrices[model_number][current_data_type]['n_distinct'] = n_distinct

            if adm_changed is True:      
                count += 1

        if adm_changed is False:      
            count += 1

for model_number in model_numbers:
    file_txt = 'Testing/custom_filtering/output_lepard_default_kpfcn_model_' + model_number + '.txt'
    file_txt = open(file_txt, 'r')
    Lines = file_txt.readlines()
    current_data_type = ''
    for line in Lines:
        if line[:-1] in data_types:
            current_data_type = line[:-1]

        if 'number of true landmarks correspondences returned from Lepard' in line:
            search = list(map(int, re.findall(r'\d+', line)))
            true = int(search[0])
            total = int(search[1])
            if adm_changed is True:
                final_matrices[model_number][current_data_type]['lepard']['true'] = true
                final_matrices[model_number][current_data_type]['lepard']['total'] = total - true
            else:
                final_matrices[model_number][current_data_type]['lepard']['true'] = true
                final_matrices[model_number][current_data_type]['lepard']['total'] = total - true

        if 'RMSE' in line:
            rmse = float(re.findall("\d+\.\d+", line)[0])
            if adm_changed is True:
                final_matrices[model_number][current_data_type]['lepard']['rmse'] = rmse
            else:
                final_matrices[model_number][current_data_type]['lepard']['rmse'] = rmse

for model_number in model_numbers:
    file_txt = 'Testing/custom_filtering/output_outlier_rejection_default_kpfcn_model_' + model_number + '.txt'
    file_txt = open(file_txt, 'r')
    Lines = file_txt.readlines()
    current_data_type = ''
    count = 0
    for line in Lines:
        if line[:-1] in data_types:
            current_data_type = line[:-1]

        if 'number of true landmark correspondences returned from Outlier Rejection' in line:
            search = list(map(int, re.findall(r'\d+', line)))
            true = int(search[0])
            total = int(search[1])
            if adm_changed is True:
                final_matrices[model_number][current_data_type]['outlier']['true'] = true
                final_matrices[model_number][current_data_type]['outlier']['total'] = total - true
            else:
                final_matrices[model_number][current_data_type]['outlier']['true'] = true
                final_matrices[model_number][current_data_type]['outlier']['total'] = total - true
        
        if 'RMSE' in line:
            rmse = float(re.findall("\d+\.\d+", line)[0])
            if adm_changed is True:
                final_matrices[model_number][current_data_type]['outlier']['rmse'] = rmse
            else:
                final_matrices[model_number][current_data_type]['outlier']['rmse'] = rmse

 # Changing ADM
'''
modified_adm = ['custom - ' + str(adm_r) for adm_r in adm]
modified_adm_pos = range(len(modified_adm))

plt.title(data_type + ' - varying ADM')
plt.plot(modified_adm_pos, rmse, color='r')
plt.xticks(modified_adm_pos, modified_adm, rotation=90)
plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_rmse_nc_' + str(nc[0]) + '_iot_' + str(iot[0]) + '_sampling_' + sampling + '_varying_adm.png', bbox_inches='tight')

modified_adm_lepard_outlier = ['custom - ' + str(adm_r) for adm_r in adm]
modified_adm_lepard_outlier.append('lepard')
modified_adm_lepard_outlier.append('outlier rejection')
modified_adm_lepard_outlier_pos = range(len(modified_adm_lepard_outlier))

plt.title(data_type + ' - varying ADM')
plt.bar(modified_adm_lepard_outlier_pos, true_data, color='r')
plt.bar(modified_adm_lepard_outlier_pos, total_data, bottom=true_data, color='b')
plt.xticks(modified_adm_lepard_outlier_pos, modified_adm_lepard_outlier, rotation=90)
plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_gt_ratio_barchart_nc_' + str(nc[0]) + '_iot_' + str(iot[0]) + '_sampling_' + sampling + '_varying_adm.png', bbox_inches='tight')

plt.clf()
plt.title(data_type + ' - varying ADM')
plt.plot(modified_adm_lepard_outlier_pos, fraction, color='r')
plt.xticks(modified_adm_lepard_outlier_pos, modified_adm_lepard_outlier, rotation=90)
plt.ylim(0, 1)
plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_gt_ratio_graph_nc_' + str(nc[0]) + '_iot_' + str(iot[0]) + '_sampling_' + sampling + '_varying_adm.png', bbox_inches='tight')
'''

for data_type in data_types:

     # Changing NC
    modified_nc = [str(nc_v) for nc_v in nc]
    modified_nc.append('lepard')
    modified_nc.append('outlier rejection')
    modified_nc_pos = range(len(modified_nc))
    
    plt.clf()

    for model_number in model_numbers:
        
        true_data = []
        total_data = []
        
        for i in range(len(nc)) :
            for j in range(len(adm)):
                for k in range(len(iot)):
                        
                    true_data.append(final_matrices[model_number][data_type]['custom']['true'][i][j][k])
                    total_data.append(final_matrices[model_number][data_type]['custom']['total'][i][j][k])
            
        true_data.append(final_matrices[model_number][data_type]['lepard']['true'])
        total_data.append(final_matrices[model_number][data_type]['lepard']['total'])

        true_data.append(final_matrices[model_number][data_type]['outlier']['true'])
        total_data.append(final_matrices[model_number][data_type]['outlier']['total'])

        plt.clf()
        plt.title(data_type)
        plt.bar(modified_nc_pos, true_data, color='r')
        plt.bar(modified_nc_pos, total_data, bottom=true_data, color='b')
        plt.xticks(modified_nc_pos, modified_nc_pos, rotation=90)
        plt.axhline(y = final_matrices[model_number][data_type]['n_distinct'], color = 'r', linestyle = '-')
        plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_gt_ratio_barchart_adm_' + str(adm[0]) + '_iot_' + str(iot[0]) + '_sampling_' + sampling + '_' + type + '_model_' + model_number + '_varying_nc.png', bbox_inches='tight')
    
    plt.clf()
    for model_number in model_numbers:
        
        rmse = []
        
        for i in range(len(nc)) :
            for j in range(len(adm)):
                for k in range(len(iot)):

                    rmse.append(final_matrices[model_number][data_type]['custom']['rmse'][i][j][k])
            
        rmse.append(final_matrices[model_number][data_type]['lepard']['rmse'])
        rmse.append(final_matrices[model_number][data_type]['outlier']['rmse'])

        plt.plot(modified_nc_pos, rmse, label = model_number)

    plt.title(data_type)
    plt.legend(loc ="upper right")
    plt.xticks(modified_nc_pos, modified_nc, rotation=90)
    plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_rmse_adm_' + str(adm[0]) + '_iot_' + str(iot[0]) + '_sampling_' + sampling + '_' + type + '_varying_nc.png', bbox_inches='tight')
        
        