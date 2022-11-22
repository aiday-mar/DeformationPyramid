import re

class File:
    def __init__(self, type, preprocessing, confidence, coarse_level, index_coarse):
        self.type = type
        self.preprocessing = preprocessing
        self.confidence = confidence
        self.coarse_level = coarse_level
        self.index_coarse = index_coarse
         
files=[
    "type_custom_preprocessing_mutual_confidence_0.1_coarse_level_-2_index_coarse_1.txt", 
    "type_custom_preprocessing_none_confidence_0.1_coarse_level_-2_index_coarse_1.txt", 
    "type_custom_preprocessing_none_confidence_0.1_coarse_level_-3_index_coarse_1.txt", 
    "type_custom_preprocessing_none_confidence_0.1_coarse_level_-3_index_coarse_2.txt",
    "type_custom_preprocessing_none_confidence_0.05_coarse_level_-2_index_coarse_1.txt",
    "type_default_preprocessing_mutual_confidence_0.1_coarse_level_-2_index_coarse_1.txt"
]
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

print('final_data : ', final_data)
RMSE_full_deformed = []
RMSE_full_non_deformed = []
RMSE_partial_deformed = []
RMSE_partial_non_deformed = []

for file in files:
    RMSE_full_deformed.append(final_data[file]['Full Deformed']['RMSE'])
    RMSE_full_non_deformed.append(final_data[file]['Full Non Deformed']['RMSE'])
    RMSE_partial_deformed.append(final_data[file]['Partial Deformed']['RMSE'])
    RMSE_partial_non_deformed.append(final_data[file]['Partial Non Deformed']['RMSE'])
    
print('RMSE_full_deformed : ', RMSE_full_deformed)
print('RMSE_full_non_deformed : ', RMSE_full_non_deformed)
print('RMSE_partial_deformed : ', RMSE_partial_deformed)
print('RMSE_partial_non_deformed : ', RMSE_partial_non_deformed)