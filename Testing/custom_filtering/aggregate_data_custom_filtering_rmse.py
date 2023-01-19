import re
import matplotlib.pyplot as plt
import numpy as np
import copy

data_types=['Full Deformed', 'Partial Deformed']
model_numbers = ['002', '042', '085', '126', '167', '207']

preprocessing = 'mutual'
nc=150
adm=3.0

if preprocessing == 'mutual':
    weights = {
        'full_deformed' : {
            'kpfcn' : {
                'conf' : '0.000001',
            },
            'fcgf' : {
                'conf' : '0.000001',
            }
        },
        'partial_deformed' : {
            'kpfcn' : {
                'conf' : '0.000001',
            },
            'fcgf' : {
                'conf' : '0.000001',
            }
        }   
    }
elif preprocessing == 'none':
    weights = {
        'full_deformed' : {
            'kpfcn' : {
                'conf' : '0.000001',
            },
            'fcgf' : {
                'conf' : '0.000001',
            }
        },
        'partial_deformed' : {
            'kpfcn' : {
                'conf' : '0.000001',
            },
            'fcgf' : {
                'conf' : '0.000001',
            }
        }   
    }
else:
    raise Exception('Must be one of the preprocessing options')

confidence = '1e-06'
sampling='linspace'
max_ldmks = 'None'

def create_final_matrix(nc):

    shape=(len(nc), len(adm), len(iot))

    final_submatrix ={
    'Full Deformed': {'lepard' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'outlier' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}, 'n_distinct' : 0}, 
    'Partial Deformed': {'lepard' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'outlier' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}, 'n_distinct' : 0},  
    }

    final_matrix = { model_number : copy.deepcopy(final_submatrix) for model_number in model_numbers}
    return final_matrix

final_matrices = {}

for model_number in model_numbers:
    final_matrices[model_number] = {}

    for training_data in weights:
        final_matrices[model_number][training_data] = {}

        for feature_extractor in weights[training_data]:
            final_matrices[model_number][training_data][feature_extractor] = {}
            final_matrices[model_number][training_data][feature_extractor]['initial'] = 0
            final_matrices[model_number][training_data][feature_extractor]['final'] = 0

print(final_matrices)

'''
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
                final_matrices[current_data_type][0][count] = rmse
        
        count += 1

# print('final_data : ', final_data)
# print('final_matrices : ', final_matrices)

# Heatmap
if heatmap:
    for data_type in data_types:
        ax = sns.heatmap(final_matrices[data_type], linewidth=0.5)
        figure = ax.get_figure()    
        figure.savefig('plots/custom_filtering_v2/' + data_type.replace(' ', '_') + '.png', dpi=400)

if not heatmap:
    for data_type in data_types:
        plt.plot(adm, np.squeeze(final_matrices[data_type].T))
    plt.legend(data_types)
    plt.savefig('plots/custom_filtering_v2/varying_radius.png')

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
'''