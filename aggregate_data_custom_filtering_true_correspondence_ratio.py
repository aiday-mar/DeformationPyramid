import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class File:
    def __init__(self, type, preprocessing, confidence, number_centers, average_distance_multiplier, coarse_level, index_coarse, number_iterations, inlier_outlier_thr):
        self.type = type
        self.preprocessing = preprocessing
        self.confidence = confidence
        self.number_centers = number_centers
        self.average_distance_multiplier = average_distance_multiplier
        self.coarse_level = coarse_level
        self.index_coarse = index_coarse
        self.number_iterations = number_iterations
        self.inlier_outlier_thr = inlier_outlier_thr
    
    def __str__(self):
        return " - Type : " + str(self.type) + " - Preprocessing : " + str(self.preprocessing) + " - Confidence : " + str(self.confidence) + " - Number Centers : " + str(self.number_centers) + " - Average Distance Multiplier : " + str(self.average_distance_multiplier) + " - Coarse Level : " + str(self.coarse_level) + " - Index Coarse : " + str(self.index_coarse) + ' - Number Iterations : ' + str(self.number_iterations) + ' - Inlier Outlier Threshold : ' + str(self.inlier_outlier_thr)

files=[]
file_types=[]
number_iterations=1
data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'TestData/'

# nc = [100, 200, 300]:
nc = [20]

# adm = [1, 2, 3, 4]
# adm =  [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0]
adm = [3.0]

iot=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

shape=(len(nc), len(adm), len(iot))
final_matrices={'Full Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'outlier' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}}, 
                'Full Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'outlier' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}}, 
                'Partial Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'outlier' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}},  
                'Partial Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'outlier' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}}}

for i in nc :
    for j in adm:
        count = 0
        for k in iot:
            file = 'v_2_t_custom_p_none_c_0.1_nc_' + str(i) + '_adm_' + str(j) + '_cl_-2_ic_1_ni_' + str(number_iterations) + '_iot_' + str(k) + '.txt'
            files.append(file)
            file_types.append(File('custom', 'none', 0.1, i, j, -2, 1, number_iterations, k))
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
                
                if 'number of true landmark correspondences returned from outlier rejection' in line:
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
    
    for i in range(len(iot)):
        true_data.append(final_matrices[data_type]['custom']['true'][0][0][i])
        total_data.append(final_matrices[data_type]['custom']['total'][0][0][i])
        rmse.append(final_matrices[data_type]['custom']['rmse'][0][0][i])
        
        if final_matrices[data_type]['custom']['total'][0][0][i] != 0:
            fraction.append(final_matrices[data_type]['custom']['true'][0][0][i]/(final_matrices[data_type]['custom']['total'][0][0][i]+final_matrices[data_type]['custom']['true'][0][0][i]))
        else:
            fraction.append(0)
            
        if i==len(iot) -1:
            true_data.append(final_matrices[data_type]['lepard']['true'][0][0][i])
            total_data.append(final_matrices[data_type]['lepard']['total'][0][0][i])
            
            true_data.append(final_matrices[data_type]['outlier']['true'][0][0][i])
            total_data.append(final_matrices[data_type]['outlier']['total'][0][0][i])
            
            if final_matrices[data_type]['lepard']['total'][0][0][i] != 0:
                fraction.append(final_matrices[data_type]['lepard']['true'][0][0][i]/(final_matrices[data_type]['lepard']['total'][0][0][i]+final_matrices[data_type]['lepard']['true'][0][0][i]))
            else:
                fraction.append(0)

            if final_matrices[data_type]['outlier']['total'][0][0][i] != 0:
                fraction.append(final_matrices[data_type]['outlier']['true'][0][0][i]/(final_matrices[data_type]['outlier']['total'][0][0][i]+final_matrices[data_type]['outlier']['true'][0][0][i]))
            else:
                fraction.append(0)
    
    '''
    modified_adm = ['custom - ' + str(adm_r) for adm_r in adm]
    modified_adm_pos = range(len(modified_adm))
    
    plt.title(data_type + ' - RMSE varying radii')
    plt.plot(modified_adm_pos, rmse, color='r')
    plt.xticks(modified_adm_pos, modified_adm, rotation=90)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_graph_rmse_for_varying_radii.png', bbox_inches='tight')
    
    modified_adm_lepard_outlier = ['custom - ' + str(adm_r) for adm_r in adm]
    modified_adm_lepard_outlier.append('lepard')
    modified_adm_lepard_outlier.append('outlier rejection')
    modified_adm_lepard_outlier_pos = range(len(modified_adm_lepard_outlier))
    
    plt.title(data_type + ' - GT ratio varying radii')
    plt.bar(modified_adm_lepard_outlier_pos, true_data, color='r')
    plt.bar(modified_adm_lepard_outlier_pos, total_data, bottom=true_data, color='b')
    plt.xticks(modified_adm_lepard_outlier_pos, modified_adm_lepard_outlier, rotation=90)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_bar_chart_true_correspondence_ratio_for_varying_radii.png', bbox_inches='tight')
    
    plt.clf()
    plt.title(data_type + ' - GT ratio varying radii')
    plt.plot(modified_adm_lepard_outlier_pos, fraction, color='r')
    plt.xticks(modified_adm_lepard_outlier_pos, modified_adm_lepard_outlier, rotation=90)
    plt.ylim(0, 1)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_graph_true_correspondence_ratio_for_varying_radii.png', bbox_inches='tight')
    '''
    
    modified_iot = [str(iot_v) for iot_v in iot]
    modified_iot_pos = range(len(modified_iot))
    
    plt.title(data_type + ' - RMSE varying inlier/outlier thresholds')
    plt.plot(modified_iot_pos, rmse, color='r')
    plt.xticks(modified_iot_pos, modified_iot, rotation=90)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_graph_rmse_for_varying_inlier_outlier_thresholds.png', bbox_inches='tight')
    
    modified_iot_lepard_outlier = [str(iot_v) for iot_v in iot]
    modified_iot_lepard_outlier.append('lepard')
    modified_iot_lepard_outlier.append('outlier rejection')
    modified_iot_lepard_outlier_pos = range(len(modified_iot_lepard_outlier))
    
    plt.title(data_type + ' - GT ratio varying inlier/outlier thresholds')
    plt.bar(modified_iot_lepard_outlier_pos, true_data, color='r')
    plt.bar(modified_iot_lepard_outlier_pos, total_data, bottom=true_data, color='b')
    plt.xticks(modified_iot_lepard_outlier_pos, modified_iot_lepard_outlier, rotation=90)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_bar_chart_true_correspondence_ratio_for_varying_inlier_outlier_thresholds.png', bbox_inches='tight')
    
    plt.clf()
    plt.title(data_type + ' - GT ratio varying radii')
    plt.plot(modified_iot_lepard_outlier_pos, fraction, color='r')
    plt.xticks(modified_iot_lepard_outlier_pos, modified_iot_lepard_outlier_pos, rotation=90)
    plt.ylim(0, 1)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_graph_true_correspondence_ratio_for_varying_inlier_outlier_thresholds.png', bbox_inches='tight')
    