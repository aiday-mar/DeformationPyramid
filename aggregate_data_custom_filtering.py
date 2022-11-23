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

# Version 2 files
files=[]
file_types=[]
number_iterations=1
data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'TestData/'

# For heatmap
# nc = [100, 200, 300]:
nc = [20]

# radius adm
# adm = [1, 2, 3, 4]
adm =  [1.0, 1.4, 1.8, 2.2, 2.6, 3.0]

# heatmap = True
heatmap = False

shape=(len(nc), len(adm))
final_matrices={'Full Non Deformed': np.zeros(shape), 'Full Deformed': np.zeros(shape), 'Partial Deformed': np.zeros(shape),  'Partial Non Deformed': np.zeros(shape)}
final_data = {}

for i in nc : 
    count = 0
    for j in adm:
        file = 'v_2_t_custom_p_none_c_0.1_nc_' + str(i) + '_adm_' + str(j) + '_cl_-2_ic_1_ni_' + str(number_iterations) +'.txt'
        files.append(file)
        file_types.append(File('custom', 'none', 0.1, i, j, -2, 1, number_iterations))
        final_data[file] = {}
        file_txt = open(base + file, 'r')
        Lines = file_txt.readlines()
        current_data_type = ''
        for line in Lines:
            if line[:-1] in data_types:
                current_data_type = line[:-1]
                final_data[file][current_data_type] = {}

            if heatmap and 'RMSE' in line:
                rmse = float(re.findall("\d+\.\d+", line)[0])
                final_data[file][current_data_type]['RMSE'] = rmse
                final_matrices[current_data_type][int(i/100)-1][j-1] = rmse

            if not heatmap and 'RMSE' in line:
                rmse = float(re.findall("\d+\.\d+", line)[0])
                final_data[file][current_data_type]['RMSE'] = rmse
                final_matrices[current_data_type][count] = rmse
        
        count += 1

print('final_data : ', final_data)
print('final_matrices : ', final_matrices)

# Heatmap
if heatmap:
    for data_type in data_types:
        ax = sns.heatmap(final_matrices[data_type], linewidth=0.5)
        figure = ax.get_figure()    
        figure.savefig('plots/custom_filtering_v2/' + data_type.replace(' ', '_') + '.png', dpi=400)

if not heatmap:
    for data_type in data_types:
        plt.plot(final_matrices[data_type])
        plt.savefig('plots/custom_filtering_v2/' + data_type.replace(' ', '_') + '_varying_radius_' + '.png')

RMSE_full_deformed = []
RMSE_full_non_deformed = []
RMSE_partial_deformed = []
RMSE_partial_non_deformed = []

for file in files:
    RMSE_full_deformed.append(final_data[file]['Full Deformed']['RMSE'])
    RMSE_full_non_deformed.append(final_data[file]['Full Non Deformed']['RMSE'])
    RMSE_partial_deformed.append(final_data[file]['Partial Deformed']['RMSE'])
    RMSE_partial_non_deformed.append(final_data[file]['Partial Non Deformed']['RMSE'])

fig = plt.figure(figsize = (10, 5))
plt.bar(files, RMSE_full_deformed, width = 0.4)
plt.ylabel("RMSE")
plt.title("RMSE - Full Deformed")
plt.savefig('plots/custom_filtering_v2/RMSE_Full_Deformed_Custom_Filtering.png')

fig = plt.figure(figsize = (10, 5))
plt.bar(files, RMSE_full_non_deformed, width = 0.4)
plt.ylabel("RMSE")
plt.title("RMSE - Full Non Deformed")
plt.savefig('plots/custom_filtering_v2/RMSE_Full_Non_Deformed_Custom_Filtering.png')

fig = plt.figure(figsize = (10, 5))
plt.bar(files, RMSE_partial_deformed, width = 0.4)
plt.ylabel("RMSE")
plt.title("RMSE - Partial Deformed")
plt.savefig('plots/custom_filtering_v2/RMSE_Partial_Deformed_Custom_Filtering.png')

fig = plt.figure(figsize = (10, 5))
plt.bar(files, RMSE_partial_non_deformed, width = 0.4)
plt.ylabel("RMSE")
plt.title("RMSE - Partial Non Deformed")
plt.savefig('plots/custom_filtering_v2/RMSE_Partial_Non_Deformed_Custom_Filtering.png')

print('RMSE_full_deformed : ', RMSE_full_deformed)
RMSE_full_deformed_index = RMSE_full_deformed.index(min(RMSE_full_deformed))
print('Minimum attained for : ', file_types[RMSE_full_deformed_index])

print('RMSE_full_non_deformed : ', RMSE_full_non_deformed)
RMSE_full_non_deformed_index = RMSE_full_non_deformed.index(min(RMSE_full_non_deformed))
print('Minimum attained for : ', file_types[RMSE_full_non_deformed_index])

print('RMSE_partial_deformed : ', RMSE_partial_deformed)
RMSE_partial_deformed_index = RMSE_partial_deformed.index(min(RMSE_partial_deformed))
print('Minimum attained for : ', file_types[RMSE_partial_deformed_index])

print('RMSE_partial_non_deformed : ', RMSE_partial_non_deformed)
RMSE_partial_non_deformed_index = RMSE_partial_non_deformed.index(min(RMSE_partial_non_deformed))
print('Minimum attained for : ', file_types[RMSE_partial_non_deformed_index])