import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'Testing/custom_filtering/'

# nc = [100, 200, 300]:
nc = [1, 5, 10, 50, 100, 200]

# adm = [1, 2, 3, 4]
# adm =  [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0]
adm = [3.0]

# iot=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
# iot = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
iot = [0.01]

# sampling = 'poisson'
sampling='linspace'

version=4

shape=(len(nc), len(adm), len(iot))
final_matrices={'Full Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'outlier' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}}, 
                'Full Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'outlier' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}}, 
                'Partial Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'outlier' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}},  
                'Partial Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'outlier' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}}}

for i in nc :
    for j in adm:
        count = 0
        for k in iot:
            file = 'v_' + str(version) + '_t_custom_p_none_c_0.1_nc_' + str(i) + '_adm_' + str(j) + '_cl_-2_ic_1_ni_' + str(number_iterations) + '_iot_' + str(k) + '_s_' + sampling + '.txt'
            files.append(file)
            file_types.append(File('custom', 'none', 0.1, i, j, -2, 1, number_iterations, k, sampling))
            file_txt = open(base + file, 'r')
            Lines = file_txt.readlines()
            current_data_type = ''
            for line in Lines:
                if line[:-1] in data_types:
                    current_data_type = line[:-1]

                if 'number of true landmarks correspondences returned from Lepard' in line:
                    search = list(map(int, re.findall(r'\d+', line)))
                    true = int(search[0])
                    total = int(search[1])
                    final_matrices[current_data_type]['lepard']['true'][0][0][count] = true
                    final_matrices[current_data_type]['lepard']['total'][0][0][count] = total - true
                
                if 'number of true landmark correspondences returned from custom filtering' in line:
                    search = list(map(int, re.findall(r'\d+', line)))
                    true = int(search[0])
                    total = int(search[1])
                    final_matrices[current_data_type]['custom']['true'][0][0][count] = true
                    final_matrices[current_data_type]['custom']['total'][0][0][count] = total - true
                
                if 'number of true landmark correspondences returned from Outlier Rejection' in line:
                    search = list(map(int, re.findall(r'\d+', line)))
                    true = int(search[0])
                    total = int(search[1])
                    final_matrices[current_data_type]['outlier']['true'][0][0][count] = true
                    final_matrices[current_data_type]['outlier']['total'][0][0][count] = total - true
                
                if 'RMSE' in line:
                    rmse = float(re.findall("\d+\.\d+", line)[0])
                    final_matrices[current_data_type]['custom']['rmse'][0][0][count] = rmse
            
            count += 1

for data_type in data_types:
    plt.clf()
    true_data = []
    rmse = []
    total_data = []
    fraction = []
    
    for i in range(len(nc)) :
        for j in range(len(adm)):
            for k in range(len(iot)):
                    
                true_data.append(final_matrices[data_type]['custom']['true'][i][j][k])
                total_data.append(final_matrices[data_type]['custom']['total'][i][j][k])
                rmse.append(final_matrices[data_type]['custom']['rmse'][i][j][k])
                
                if final_matrices[data_type]['custom']['total'][i][j][k] != 0:
                    fraction.append(final_matrices[data_type]['custom']['true'][i][j][k]/(final_matrices[data_type]['custom']['total'][i][j][k]+final_matrices[data_type]['custom']['true'][i][j][k]))
                else:
                    fraction.append(0)
                    
                if k==len(iot) -1:
                    true_data.append(final_matrices[data_type]['lepard']['true'][i][j][k])
                    total_data.append(final_matrices[data_type]['lepard']['total'][i][j][k])
                    
                    true_data.append(final_matrices[data_type]['outlier']['true'][i][j][k])
                    total_data.append(final_matrices[data_type]['outlier']['total'][i][j][k])
                    
                    if final_matrices[data_type]['lepard']['total'][i][j][k] != 0:
                        fraction.append(final_matrices[data_type]['lepard']['true'][i][j][k]/(final_matrices[data_type]['lepard']['total'][i][j][k]+final_matrices[data_type]['lepard']['true'][i][j][k]))
                    else:
                        fraction.append(0)

                    if final_matrices[data_type]['outlier']['total'][i][j][k] != 0:
                        fraction.append(final_matrices[data_type]['outlier']['true'][i][j][k]/(final_matrices[data_type]['outlier']['total'][i][j][k]+final_matrices[data_type]['outlier']['true'][i][j][k]))
                    else:
                        fraction.append(0)
            
    # Changing IOT
    '''    
    modified_iot = [str(iot_v) for iot_v in iot]
    modified_iot_pos = range(len(modified_iot))
                
    plt.title(data_type + ' - RMSE - nc : ' + str(nc[i]) + ' - adm : ' + str(adm[j]) + ' - sampling : ' + sampling + ' varying I/O threshold')
    plt.plot(modified_iot_pos, rmse, color='r')
    plt.xticks(modified_iot_pos, modified_iot, rotation=90)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_rmse_nc_' + str(nc[i]) + '_adm_' + str(adm[j]) + '_sampling_' + sampling + '_varying_iot.png', bbox_inches='tight')
    
    modified_iot_lepard_outlier = [str(iot_v) for iot_v in iot]
    modified_iot_lepard_outlier.append('lepard')
    modified_iot_lepard_outlier.append('outlier rejection')
    modified_iot_lepard_outlier_pos = range(len(modified_iot_lepard_outlier))
    
    plt.title(data_type + ' - GT ratio - nc : ' + str(nc[i]) + ' - adm : ' + str(adm[j]) + ' - sampling : ' + sampling + ' varying I/O threshold')
    plt.bar(modified_iot_lepard_outlier_pos, true_data, color='r')
    plt.bar(modified_iot_lepard_outlier_pos, total_data, bottom=true_data, color='b')
    plt.xticks(modified_iot_lepard_outlier_pos, modified_iot_lepard_outlier, rotation=90)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_gt_ratio_barchart_nc_' + str(nc[i]) + '_adm_' + str(adm[j]) + '_sampling_' + sampling + '_varying_iot.png', bbox_inches='tight')
    
    plt.clf()
    plt.title(data_type + ' - GT ratio - nc : ' + str(nc[i]) + ' - adm : ' + str(adm[j]) + ' - sampling : ' + sampling + ' varying I/O threshold')
    plt.plot(modified_iot_lepard_outlier_pos, fraction, color='r')
    plt.xticks(modified_iot_lepard_outlier_pos, modified_iot_lepard_outlier_pos, rotation=90)
    plt.ylim(0, 1)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_gt_ratio_graph_nc_' + str(nc[i]) + '_adm_' + str(adm[j]) + '_sampling_' + sampling + '_varying_iot.png', bbox_inches='tight')
    '''
    
    # Changing ADM
    '''
    modified_adm = ['custom - ' + str(adm_r) for adm_r in adm]
    modified_adm_pos = range(len(modified_adm))
    
    plt.title(data_type + ' - RMSE - nc : ' + str(nc[i]) + ' - sampling : ' + sampling + ' varying ADM')
    plt.plot(modified_adm_pos, rmse, color='r')
    plt.xticks(modified_adm_pos, modified_adm, rotation=90)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_rmse_nc_' + str(nc[i]) + '_adm_' + str(adm[j]) + '_sampling_' + sampling + '_varying_adm.png', bbox_inches='tight')
    
    modified_adm_lepard_outlier = ['custom - ' + str(adm_r) for adm_r in adm]
    modified_adm_lepard_outlier.append('lepard')
    modified_adm_lepard_outlier.append('outlier rejection')
    modified_adm_lepard_outlier_pos = range(len(modified_adm_lepard_outlier))
    
    plt.title(data_type + ' - GT ratio - nc : ' + str(nc[i]) + ' - sampling : ' + sampling + ' varying ADM')
    plt.bar(modified_adm_lepard_outlier_pos, true_data, color='r')
    plt.bar(modified_adm_lepard_outlier_pos, total_data, bottom=true_data, color='b')
    plt.xticks(modified_adm_lepard_outlier_pos, modified_adm_lepard_outlier, rotation=90)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_rmse_nc_' + str(nc[i]) + '_adm_' + str(adm[j]) + '_sampling_' + sampling + '_varying_adm.png', bbox_inches='tight')
    
    plt.clf()
    plt.title(data_type + ' - GT ratio - nc : ' + str(nc[i]) + ' - sampling : ' + sampling + ' varying ADM')
    plt.plot(modified_adm_lepard_outlier_pos, fraction, color='r')
    plt.xticks(modified_adm_lepard_outlier_pos, modified_adm_lepard_outlier, rotation=90)
    plt.ylim(0, 1)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_rmse_nc_' + str(nc[i]) + '_adm_' + str(adm[j]) + '_sampling_' + sampling + '_varying_adm.png', bbox_inches='tight')
    '''  
            
    # Changing NC
    modified_nc = [str(nc_v) for nc_v in nc]
    modified_nc_pos = range(len(modified_nc))
                
    plt.title(data_type + ' - RMSE - adm : ' + str(adm[0]) + ' - sampling : ' + sampling + ' varying number of centers')
    plt.plot(modified_nc_pos, rmse, color='r')
    plt.xticks(modified_nc_pos, modified_nc, rotation=90)
    plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_rmse_adm_' + str(adm[0]) + '_iot_' + str(iot[0]) + '_sampling_' + sampling + '_varying_nc.png', bbox_inches='tight')
    
    modified_nc_lepard_outlier = [str(nc_v) for nc_v in nc]
    modified_nc_lepard_outlier.append('lepard')
    modified_nc_lepard_outlier.append('outlier rejection')
    modified_nc_lepard_outlier_pos = range(len(modified_nc_lepard_outlier))
    
    plt.title(data_type + ' - GT ratio - adm : ' + str(adm[0]) + ' - sampling : ' + sampling + ' varying number of centers')
    plt.bar(modified_nc_lepard_outlier_pos, true_data, color='r')
    plt.bar(modified_nc_lepard_outlier_pos, total_data, bottom=true_data, color='b')
    plt.xticks(modified_nc_lepard_outlier_pos, modified_nc_lepard_outlier_pos, rotation=90)
    plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_gt_ratio_barchart_adm_' + str(adm[0]) + '_iot_' + str(iot[0]) + '_sampling_' + sampling + '_varying_nc.png', bbox_inches='tight')
    
    plt.clf()
    plt.title(data_type + ' - GT ratio - adm : ' + str(adm[0]) + ' - sampling : ' + sampling + ' varying number of centers')
    plt.plot(modified_nc_lepard_outlier_pos, fraction, color='r')
    plt.xticks(modified_nc_lepard_outlier_pos, modified_nc_lepard_outlier, rotation=90)
    plt.ylim(0, 1)
    plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_gt_ratio_graph_adm_' + str(adm[0]) + '_iot_' + str(iot[0]) + '_sampling_' + sampling + '_varying_nc.png', bbox_inches='tight')