import re
import matplotlib.pyplot as plt
import numpy as np
import copy

# Replaces the code in the all folder which also calculates the RMSE

data_types=['Full Deformed', 'Partial Deformed']
model_numbers = ['002', '042', '085', '126', '167', '207']
base = 'Testing/custom_filtering/'

preprocessing = 'mutual'
iot=0.01

barWidth = 0.30
barWidthPlot = 0.30
br1 = np.array([0, 1, 2, 3, 4, 5])
br2 = np.array([x + barWidth for x in br1])
bars = [br1, br2]

if preprocessing == 'mutual':
    weights = {
        'full_deformed' : {
            'kpfcn' : {
                'conf' : '0.000001',
                'adm' : '2.0',
                'nc' : '15'
            },
            'fcgf' : {
                'conf' : '0.000001',
                'adm' : '5.0',
                'nc' : '150'
            }
        },
        'partial_deformed' : {
            'kpfcn' : {
                'conf' : '0.000001',
                'adm' : '2.0',
                'nc' : '15'
            },
            'fcgf' : {
                'conf' : '0.000001',
                'adm' : '5.0',
                'nc' : '150'
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

sampling='linspace'
max_ldmks = 'None'

final_matrices = {}

for model_number in model_numbers:
    final_matrices[model_number] = {}

    for training_data in weights:
        final_matrices[model_number][training_data] = {}

        for feature_extractor in weights[training_data]:
            final_matrices[model_number][training_data][feature_extractor] = {}
            final_matrices[model_number][training_data][feature_extractor]['initial'] = 0
            final_matrices[model_number][training_data][feature_extractor]['final'] = 0

for training_data in weights:
    for feature_extractor in weights[training_data]:
        for model_number in model_numbers:
            
            nc = weights[training_data][feature_extractor]['nc']
            adm = weights[training_data][feature_extractor]['adm']
            confidence = weights[training_data][feature_extractor]['conf']

            file = 'p_' + preprocessing + '_c_' + str(confidence) + '_nc_' + str(nc) + '_adm_' + str(adm) + '_iot_' + str(iot) + '_s_' + sampling + '_max_ldmks_' +  max_ldmks + '_' + feature_extractor + '_td_' + training_data + '_model_' + model_number + '.txt'
            file_txt = open(base + file, 'r')
            Lines = file_txt.readlines()
            current_data_type = ''
            for line in Lines:

                if 'RMSE' in line:
                    rmse = float(re.findall("\d+\.\d+", line)[0])
                    final_matrices[model_number][training_data][feature_extractor]['final'] = rmse

            file_txt = 'Testing/custom_filtering/output_outlier_rejection_default_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_model_' + model_number + '.txt'
            file_txt = open(file_txt, 'r')
            Lines = file_txt.readlines()
            current_data_type = ''
            for line in Lines:
                
                if 'RMSE' in line:
                    rmse = float(re.findall("\d+\.\d+", line)[0])
                    final_matrices[model_number][training_data][feature_extractor]['initial'] = rmse

for training_data in weights:
    plt.clf()
    count = 0
    for feature_extractor in weights[training_data]:

        initial_rmse = []
        rmse = []
        for model_number in model_numbers:
            initial_rmse.append(final_matrices[model_number][training_data][feature_extractor]['initial'])
            rmse.append(final_matrices[model_number][training_data][feature_extractor]['final'])
        
        bar = bars[count]
        weights_legend = feature_extractor + ' - ' + training_data.replace('_', ' ')
        plt.bar(bar, rmse, width = barWidthPlot, label = weights_legend) 
        plt.bar(bar, initial_rmse, width = barWidthPlot, fill = False, linestyle='dashed', label='_nolegend_')
        count += 1

    title = training_data.replace('_', ' ').title()
    plt.title(title)
    plt.legend(loc = 'upper right')
    plt.xlabel('Model number')
    plt.ylabel('RMSE')
    plt.savefig('Testing/custom_filtering/ndp_final_rmse_' + training_data + '.png', bbox_inches='tight')