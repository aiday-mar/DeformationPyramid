
import re
import matplotlib.pyplot as plt
import numpy as np
from os import path

knn_matching = 'True'
# knn_matching = 'False'

data_types = ['full_deformed', 'partial_deformed', 'full_non_deformed', 'partial_non_deformed']

barWidth = 0.15
barWidthPlot = 0.15
br1 = np.array([0, 1, 2, 3, 4, 5])
br2 = np.array([x + barWidth for x in br1])
br3 = np.array([x + barWidth for x in br2])
br4 = np.array([x + barWidth for x in br3])
br5 = np.array([x + barWidth for x in br4])

if knn_matching == 'False':
    confidence_thresholds = {
        'full_deformed' : {
            'kpfcn_pretrained' : '1e-06',
            'kpfcn_full_deformed' : '1e-06',
            'fcgf_full_deformed' : '1e-06',
            'kpfcn_partial_deformed' : '1e-06',
            'fcgf_partial_deformed' : '1e-06'
        },
        'full_non_deformed' : {
            'kpfcn_pretrained' : '1e-06',
            'kpfcn_full_deformed' : '1e-06',
            'fcgf_full_deformed' : '1e-06',
            'kpfcn_partial_deformed' : '1e-06',
            'fcgf_partial_deformed' : '1e-06'  
        },
        'partial_deformed' : {
            'kpfcn_pretrained' : '1e-06', 
            'kpfcn_full_deformed' : '1e-06',
            'fcgf_full_deformed' : '1e-06',
            'kpfcn_partial_deformed' : '1e-04',
            'fcgf_partial_deformed' : '1e-06'
        },
        'partial_non_deformed' : {
            'kpfcn_pretrained' : '1e-06',
            'kpfcn_full_deformed' : '1e-06',
            'fcgf_full_deformed' : '1e-06',
            'kpfcn_partial_deformed' : '1e-04',
            'fcgf_partial_deformed' : '1e-06'
        },
    }
elif knn_matching == 'True':
    confidence_thresholds = {
        'full_deformed' : {
            'kpfcn_pretrained' : '1e-06',
            'kpfcn_full_deformed' : '1e-06',
            'fcgf_full_deformed' : '1e-06',
            'kpfcn_partial_deformed' : '1e-06',
            'fcgf_partial_deformed' : '1e-06'
        },
        'full_non_deformed' : {
            'kpfcn_pretrained' : '1e-06',
            'kpfcn_full_deformed' : '1e-06',
            'fcgf_full_deformed' : '1e-06',
            'kpfcn_partial_deformed' : '1e-06',
            'fcgf_partial_deformed' : '1e-06'  
        },

        'partial_deformed' : {
            'kpfcn_pretrained' : '1e-06', 
            'kpfcn_full_deformed' : '1e-06',
            'fcgf_full_deformed' : '1e-06',
            'kpfcn_partial_deformed' : '1e-06',
            'fcgf_partial_deformed' : '1e-06'
        },
        'partial_non_deformed' : {
            'kpfcn_pretrained' : '1e-06',
            'kpfcn_full_deformed' : '1e-06',
            'fcgf_full_deformed' : '1e-06',
            'kpfcn_partial_deformed' : '1e-06',
            'fcgf_partial_deformed' : '1e-06'
        },
    }
else:
    raise Exception('Either true or not')

weights = {
    'fcgf' : {
        'full_deformed' : {
            'epoch' : 10,
            'bar' :  br1
        },
        'partial_deformed' : {
            'epoch' : 5, 
            'bar' : br2
        }
    }, 
    'kpfcn' : {
        'full_deformed' : {
            'epoch' : 10,
            'bar' : br3
        }, 
        'partial_deformed' : {
            'epoch' : 5,
            'bar' : br4
        },
        'pretrained' : {
            'epoch' : 'null',
            'bar' : br5
        }
    }
}

number = 0
model_numbers = ['002', '042', '085', '126', '167', '207']
adm = 2.0
with_custom = False
# with_custom = True
colors = ['blue', 'orange', 'green', 'red', 'magenta']

preprocessing_custom='none'
# preprocessing_custom='mutual'

# preprocessing_normal='mutual'
preprocessing_normal='none'

model_search = re.compile(r'model (\d+)')

def get_data(data_type, feature_extractor, training_data_type, custom = False):
    if data_type == 'full_deformed' or data_type == 'partial_deformed':
        deformed = True
    else:
        deformed = False

    conf_type = feature_extractor + '_' + training_data_type

    if confidence_thresholds[data_type][conf_type] is not None:
        conf_text = '_conf_' + confidence_thresholds[data_type][conf_type]
    else:
        conf_text = ''

    epoch = str(weights[feature_extractor][training_data_type]['epoch'])
    if custom is False:
        file_path = "Testing/current_deformation/test_astrivis_" + data_type + "_current_deformation_pre_" + preprocessing_normal + "_" + feature_extractor + "_td_" + training_data_type + "_e_" + epoch + "_knn_" + knn_matching + conf_text + ".txt"
    else:
        file_path = "Testing/current_deformation/test_astrivis_" + data_type + "_current_deformation_pre_" + preprocessing_normal + "_" + feature_extractor + "_td_" + training_data_type + "_e_" + epoch + "_custom_adm_" + str(adm) + "_knn_" + knn_matching + conf_text + ".txt"
    
    if not path.exists(file_path):
        print('Does not exist, file_path : ', file_path)
        return 'Does not exist'
    
    file = open(file_path, 'r')
    lines = file.readlines()
    data = {}
    list_keywords = ['full-epe', 'full-AccR', 'full-AccS', 'full-outlier', 'vis-epe', 'vis-AccS', 'vis-AccR', 'vis-outlier', 'RMSE', 'Relaxed IR', 'Strict IR']
    final_data = {}
    key = None

    for line in lines:
        if 'model' in line and len(line) < 35:
            if data:
                final_data[key] = data
            
            words = line.split(' ')
            model_number = words[1]
            if deformed:
                partial1 = words[3]
                partial2 = words[5]
                partial2 = partial2[:-1]
                key = model_number + '_' + partial1 + '_' + partial2
            else:
                key = model_number

            data = {}
            data['model_number'] = model_number
            if deformed:
                data['partial1'] = partial1
                data['partial2'] = partial2
        
        for keyword in list_keywords:
            if keyword in line:
                list_res = re.findall("\d+\.\d+", line)
                res = list_res[0]
                data[keyword] = res
    else:
        final_data[key] = data
    
    return final_data

for data_type in data_types:
    number += 1
    plt.clf()
    f = plt.figure(number)
    legend = []

    print('\n')
    print('data_type : ', data_type)
    if with_custom is False:
        color_idx = 0
        for feature_extractor in weights:
            for training_data_type in weights[feature_extractor]:
                
                epoch = str(weights[feature_extractor][training_data_type]['epoch'])
                bar = weights[feature_extractor][training_data_type]['bar']

                training_data_type_mod = training_data_type.replace('_', ' ')
                weights_legend = feature_extractor + ' - ' + training_data_type_mod + ' - ' + epoch 
                legend.append(weights_legend)
                data = get_data(data_type, feature_extractor, training_data_type)
                
                if data == 'Does not exist':
                    break

                rmse = []             
                for model_number in data:
                    if 'RMSE' in data[model_number]:
                        rmse.append(float(data[model_number]['RMSE']))
                    else:
                        rmse.append(np.nan)

                print('feature_extractor : ', feature_extractor)
                print('training_data_type : ', training_data_type)

                if np.isnan(np.sum(rmse)):
                    print('rmse with NaN: ', rmse)
                # plt.plot(model_numbers, rmse, color = colors[color_idx], label=weights_legend)
                plt.bar(bar, rmse, color = colors[color_idx], width = barWidthPlot, edgecolor='white', label = weights_legend)    
                
                for i in range(len(rmse)):
                    if rmse[i] is np.nan:
                        plt.axvline(x=bar[i], color='red', ls='--')
                color_idx += 1
    else:
        color_idx = 0
        for feature_extractor in weights:
            for training_data_type in weights[feature_extractor]:
                for custom in [False, True]:
                    epoch = str(weights[feature_extractor][training_data_type])
                    training_data_type_mod = training_data_type.replace('_', ' ')
                    if custom is True:
                        weights_legend = feature_extractor + ' - ' + training_data_type_mod + ' - ' + epoch + ' - custom filtering' 
                    else: 
                        weights_legend = feature_extractor + ' - ' + training_data_type_mod + ' - ' + epoch

                    legend.append(weights_legend)
                    data = get_data(data_type, feature_extractor, training_data_type, custom)

                    if data == 'Does not exist':
                        break

                    rmse = []             
                    for model_number in data:
                        rmse.append(float(data[model_number]['RMSE']))
                    
                    if custom is False:
                        plt.plot(model_numbers, rmse, color = colors[color_idx], label=weights_legend)
                    else:
                        plt.plot(model_numbers, rmse, color = colors[color_idx], linestyle='dashed', label='_nolegend_')
                
                color_idx += 1

    plt.xlabel("Model number")
    plt.ylabel("RMSE")
    plt.xticks([r + barWidth for r in range(len(model_numbers))], model_numbers)
    plt.legend(loc='upper right')
    title = data_type.replace('_', ' ')
    title = title.title()
    plt.title(title, wrap=True)

    conf_text = '_conf_' + confidence_thresholds[data_type]['kpfcn_pretrained'] + '_' + confidence_thresholds[data_type]['kpfcn_full_deformed'] + '_' + confidence_thresholds[data_type]['fcgf_full_deformed'] + '_' + confidence_thresholds[data_type]['kpfcn_partial_deformed'] + '_' + confidence_thresholds[data_type]['fcgf_partial_deformed']

    if with_custom is False:
        plt.savefig('Testing/current_deformation/per_data_type_' + data_type + '_pre_' + preprocessing_normal + '_knn_' + knn_matching  + conf_text + '_rmse.png')
    else:
        plt.savefig('Testing/current_deformation/per_data_type_' + data_type + '_pre_' + preprocessing_custom + '_knn_' + knn_matching  + conf_text + '_rmse_custom.png')
 