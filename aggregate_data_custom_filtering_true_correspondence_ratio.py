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
final_matrices={'Full Non Deformed': np.zeros(shape), 'Full Deformed': np.zeros(shape), 'Partial Deformed': np.zeros(shape),  'Partial Non Deformed': np.zeros(shape)}

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

            if 'RMSE' in line:
                rmse = float(re.findall("\d+\.\d+", line)[0])
                final_matrices[current_data_type][0][count] = rmse
        
        count += 1

for data_type in data_types:
    plt.plot(adm, np.squeeze(final_matrices[data_type].T))
    plt.legend(data_types)
    plt.savefig('plots/custom_filtering_v4/true_correspondence_ratio_for_varying_radius.png')
