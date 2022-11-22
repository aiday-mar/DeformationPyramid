import re
import matplotlib.pyplot as plt

class File:
    def __init__(self, type, preprocessing, confidence, coarse_level, index_coarse):
        self.type = type
        self.preprocessing = preprocessing
        self.confidence = confidence
        self.coarse_level = coarse_level
        self.index_coarse = index_coarse
    
    def __str__(self):
        return " - Type : " + str(self.type) + " - Preprocessing : " + str(self.preprocessing) + " - Confidence : " + str(self.confidence) + " - Coarse Level : " + str(self.coarse_level) + " - Index Coarse : " + str(self.index_coarse)

# Version 1 files
'''        
files=[
    "version_1_type_custom_preprocessing_mutual_confidence_0.1_number_centers_1000_coarse_level_-2_index_coarse_1.txt", 
    "version_1_type_custom_preprocessing_none_confidence_0.1_number_centers_1000_coarse_level_-2_index_coarse_1.txt", 
    "version_1_type_custom_preprocessing_none_confidence_0.1_number_centers_1000_coarse_level_-3_index_coarse_1.txt", 
    "version_1_type_custom_preprocessing_none_confidence_0.1_number_centers_1000_coarse_level_-3_index_coarse_2.txt",
    "version_1_type_custom_preprocessing_none_confidence_0.05_number_centers_1000_coarse_level_-2_index_coarse_1.txt",
    "version_1_type_default_preprocessing_mutual_confidence_0.1_number_centers_1000_coarse_level_-2_index_coarse_1.txt"
]
'''

# Version 2 files
files=[
    "version_1_type_custom_preprocessing_none_confidence_0.1_number_centers_1000_coarse_level_-2_index_coarse_1.txt"
]

for i in [100, 200, 300, 400]:
    for j in [1, 2, 3, 4]:
        files.append('v_2_t_custom_p_none_c_0.1_nc_' + str(i) + '_adm_' + str(j) + '_cl_-2_ic_1.txt')

file_types=[
    File('custom', 'mutual', 0.1, -2, 1),
    File('custom', 'none', 0.1, -2, 1),
    File('custom', 'none', 0.1, -3, 1),
    File('custom', 'none', 0.1, -3, 2),
    File('custom', 'none', 0.05, -2, 1),
    File('default', 'mutual', 0.1, -2, 1)
]
data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'TestData/'
final_data = {}
for file in files:
    final_data[file] = {}
    file_txt = open(base + file, 'r')
    Lines = file_txt.readlines()
    current_data_type = None
    for line in Lines:
        if line[:-1] in data_types:
            current_data_type = line[:-1]
            final_data[file][current_data_type] = {}
                    
        if 'RMSE' in line:
            final_data[file][current_data_type]['RMSE'] = float(re.findall("\d+\.\d+", line)[0])

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