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
        print('line : ', line)
        for data_type in data_types:
            if data_type in line:
                current_data_type = line
                final_data[file][current_data_type] = {}
            
        print('final_data : ', final_data)
        
        if 'RMSE' in line:
            final_data[file][current_data_type]['RMSE'] = re.findall("\d+\.\d+", line)

print('final_data : ', final_data)