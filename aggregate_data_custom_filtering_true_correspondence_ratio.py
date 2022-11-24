import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class File:
    def __init__(self, type, preprocessing, confidence, number_centers, average_distance_multiplier, coarse_level, index_coarse, number_iterations):
        self.type = type
        self.preprocessing = preprocessing
        self.confidence = confidence
        self.number_centers = number_centers
        self.average_distance_multiplier = average_distance_multiplier
        self.coarse_level = coarse_level
        self.index_coarse = index_coarse
        self.number_iterations = number_iterations
    
    def __str__(self):
        return " - Type : " + str(self.type) + " - Preprocessing : " + str(self.preprocessing) + " - Confidence : " + str(self.confidence) + " - Number Centers : " + str(self.number_centers) + " - Average Distance Multiplier : " + str(self.average_distance_multiplier) + " - Coarse Level : " + str(self.coarse_level) + " - Index Coarse : " + str(self.index_coarse) + ' - Number Iterations : ' + str(self.number_iterations)

files=[]
file_types=[]
number_iterations=1
data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'TestData/'

# nc = [100, 200, 300]:
nc = [20]

# adm = [1, 2, 3, 4]
adm =  [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0]

shape=(len(nc), len(adm))
final_matrices={'Full Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}, 
                'Full Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}, 
                'Partial Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}},  
                'Partial Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}}

for i in nc : 
    count = 0
    for j in adm:
        file = 'v_2_t_custom_p_none_c_0.1_nc_' + str(i) + '_adm_' + str(j) + '_cl_-2_ic_1_ni_' + str(number_iterations) +'.txt'
        files.append(file)
        file_types.append(File('custom', 'none', 0.1, i, j, -2, 1, number_iterations))
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
                final_matrices[current_data_type]['lepard']['true'][0][count] = true
                final_matrices[current_data_type]['lepard']['total'][0][count] = total - true
            
            if 'number of true landmark correspondences returned from custom filtering' in line:
                search = list(map(int, re.findall(r'\d+', line)))
                true = int(search[0])
                total = int(search[1])
                final_matrices[current_data_type]['custom']['true'][0][count] = true
                final_matrices[current_data_type]['custom']['total'][0][count] = total - true
        
        count += 1

print('final_matrices : ', final_matrices)

for data_type in data_types:
    plt.clf()
    true_data = []
    total_data = []
    for i in range(len(adm)):
        true_data.append(final_matrices[data_type]['custom']['true'][0][i])
        total_data.append(final_matrices[data_type]['custom']['total'][0][i])
        
        if i==len(adm) -1:
            true_data.append(final_matrices[data_type]['lepard']['true'][0][i])
            total_data.append(final_matrices[data_type]['lepard']['total'][0][i])
    
    modified_adm = ['custom - ' + str(adm_r) for adm_r in adm]
    modified_adm.append('lepard')
    print('modified_adm : ', modified_adm)
    print('true_data : ', true_data)
    print('total_data : ', total_data)
    x_pos = range(len(modified_adm))

    plt.bar(x_pos, true_data, color='r')
    plt.bar(x_pos, total_data, bottom=true_data, color='b')
    plt.xticks(x_pos, modified_adm, rotation=90)
    plt.savefig('plots/custom_filtering_v4/' + data_type.replace(' ', '_') + '_true_correspondence_ratio_for_varying_radii.png', bbox_inches='tight')