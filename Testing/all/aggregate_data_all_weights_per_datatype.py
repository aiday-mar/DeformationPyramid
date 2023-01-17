
import re
import matplotlib.pyplot as plt
from os import path
import numpy as np
import copy

data_types = ['full_deformed', 'partial_deformed', 'full_non_deformed', 'partial_non_deformed']
model_numbers = ['002', '042', '085', '126', '167', '207']

barWidth = 0.15
barWidthPlot = 0.15
br1 = np.array([0, 1, 2, 3, 4, 5])
br2 = np.array([x + barWidth for x in br1])
br3 = np.array([x + barWidth for x in br2])
br4 = np.array([x + barWidth for x in br3])
br5 = np.array([x + barWidth for x in br4])

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

knn_matching = 'False'
# knn_matching = 'True'

number = 0
adm = 2.0
with_custom = False
# with_custom = True
colors = ['blue', 'orange', 'green', 'red', 'magenta']

preprocessing_custom='none'
# preprocessing_custom='mutual'

preprocessing_normal='mutual'
# preprocessing_normal='none'

optimized=True
def get_data(data_type, feature_extractor, training_data_type, custom = False):
    if data_type == 'full_deformed' or data_type == 'partial_deformed':
        deformed = True
    else:
        deformed = False

    epoch = str(weights[feature_extractor][training_data_type]['epoch'])
    if optimized == False:
        if custom is False:
            file_path = 'Testing/all/test_astrivis_' + data_type + '_pre_' + preprocessing_normal + '_' + feature_extractor + '_td_' + training_data_type + '_e_' + epoch + '_knn_' + knn_matching + '.txt'
        else:
            file_path = 'Testing/all/test_astrivis_' + data_type + '_pre_' + preprocessing_custom + '_' + feature_extractor + '_td_' + training_data_type + '_e_' + epoch + '_custom_adm_' + str(adm) + '_knn_' + knn_matching + '.txt'
    else:
        levels = '0'
        if data_type == 'full_non_deformed' or data_type == 'partial_non_deformed':
            levels = '1'
        elif data_type == 'full_deformed' or data_type == 'partial_deformed':
            levels = '4'

        if custom is False:
            file_path = 'Testing/all/test_astrivis_' + data_type + '_pre_' + preprocessing_normal + '_' + feature_extractor + '_td_' + training_data_type + '_e_' + epoch + '_levels_' + levels + '_knn_' + knn_matching + '.txt'
        else:
            file_path = 'Testing/all/test_astrivis_' + data_type + '_pre_' + preprocessing_custom + '_' + feature_extractor + '_td_' + training_data_type + '_e_' + epoch + '_levels_' + levels + '_custom_adm_' + str(adm) + '_knn_' + knn_matching + '.txt'

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
        if 'model' in line:
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
    # fig, ax = plt.subplots()

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

                print('feature_extractor : ', feature_extractor)
                print('training_data_type : ', training_data_type)

                if data == 'Does not exist':
                    break

                rmse = []             
                for model_number in data:
                    if 'RMSE' in data[model_number]:
                        rmse.append(float(data[model_number]['RMSE']))
                    else:
                        rmse.append(np.nan)
                
                rmse = np.array(rmse)
                plt.bar(bar, rmse, color = colors[color_idx], width = barWidthPlot, edgecolor='white', label = weights_legend)              
                color_idx += 1
    else:
        color_idx = 0
        for feature_extractor in weights:
            for training_data_type in weights[feature_extractor]:
                for custom in [False, True]:
                    epoch = str(weights[feature_extractor][training_data_type]['epoch'])
                    bar = weights[feature_extractor][training_data_type]['bar']
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
                        plt.bar(bar, rmse, color = colors[color_idx], width = barWidthPlot, edgecolor='white', label = weights_legend) 
                    else:
                        # plt.bar(bar, rmse, color = colors[color_idx], width = barWidthPlot, edgecolor='white', label = weights_legend) 
                        plt.plot(model_numbers, rmse, color = colors[color_idx], linestyle='dashed', label='_nolegend_')
                
                color_idx += 1

    plt.xlabel("Model number")
    plt.ylabel("RMSE")
    plt.xticks([r + barWidth for r in range(len(model_numbers))], model_numbers)
    plt.legend(loc='upper right')
    title = data_type.replace('_', ' ')
    title = title.title()
    plt.title(title, wrap=True)
    if optimized is False:
        if with_custom is False:
            plt.savefig('Testing/all/per_data_type_' + data_type + '_pre_' + preprocessing_normal + '_knn_' + knn_matching + '_rmse.png')
        else:
            plt.savefig('Testing/all/per_data_type_' + data_type + '_pre_' + preprocessing_custom + '_knn_' + knn_matching + ' _rmse_custom.png')
    else:
        if with_custom is False:
            plt.savefig('Testing/all/per_data_type_' + data_type + '_pre_' + preprocessing_normal + '_knn_' + knn_matching + '_optimized_rmse.png')
        else:
            plt.savefig('Testing/all/per_data_type_' + data_type + '_pre_' + preprocessing_custom + '_knn_' + knn_matching + ' _optimized_rmse_custom.png')
'''
for data_type in data_types:
    number += 1
    plt.clf()
    f = plt.figure(number)
    legend = []

    for feature_extractor in weights:
        for training_data_type in weights[feature_extractor]:
            
            epoch = str(weights[feature_extractor][training_data_type])
            training_data_type_mod = training_data_type.replace('_', ' ')
            weights_legend = feature_extractor + ' - ' + training_data_type_mod + ' - ' + epoch 
            legend.append(weights_legend)
            data = get_data(data_type, feature_extractor, training_data_type)
            relaxed_ir = []             
            for model_number in data:
                relaxed_ir.append(float(data[model_number]['Relaxed IR']))
            
            plt.plot(model_numbers, relaxed_ir)
    
    plt.xlabel("Model number")
    plt.ylabel("Relaxed IR")
    plt.legend(legend, loc='upper right')
    title = data_type.replace('_', ' ')
    title = title.title()
    plt.title(title, wrap=True)
    plt.savefig('Testing/all/per_data_type_' + data_type + '_pre_' + preprocessing_normal + '_relaxed_ir.png')        
'''        