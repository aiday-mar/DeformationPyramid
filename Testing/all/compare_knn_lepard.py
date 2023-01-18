from os import path
import re
import matplotlib.pyplot as plt
import numpy as np

td = 'full_deformed'
epoch = '10'
feature_extractor = 'fcgf'
data_type_list = ['full_deformed', 'full_non_deformed']
optimized = True
preprocessing_normal = 'mutual'
preprocessing_custom = 'mutual'
knn_matching_list = ['True', 'False']
colors = ['green', 'orange']
model_numbers = ['002', '042', '085', '126', '167', '207']
barWidth = 0.40
barWidthPlot = 0.40
br1 = np.array([0, 1, 2, 3, 4, 5])
br2 = np.array([x + barWidth for x in br1])
bars = [br1, br2]

def get_data(data_type, feature_extractor, training_data_type, knn_matching, get_optimized_data = False):

    if data_type == 'full_deformed' or data_type == 'partial_deformed':
        deformed = True
    else:
        deformed = False

    if get_optimized_data is True:
        if deformed is True:
            levels = '8'
        else:
            levels = '1'
        file_path = 'Testing/all/test_astrivis_' + data_type + '_pre_' + preprocessing_custom + '_' + feature_extractor + '_td_' + training_data_type + '_e_' + epoch + '_levels_' + levels + '_knn_' + knn_matching + '.txt'
    else:
        file_path = 'Testing/all/test_astrivis_' + data_type + '_pre_' + preprocessing_normal + '_' + feature_extractor + '_td_' + training_data_type + '_e_' + epoch + '_knn_' + knn_matching + '.txt'

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

number = 0
for data_type in data_type_list:
    number += 1
    plt.clf()
    f = plt.figure(number)
    index = 0
    for knn_matching in knn_matching_list:
        data = get_data(data_type, feature_extractor, td, knn_matching, optimized)
        rmse = []             
        for model_number in data:
            rmse.append(float(data[model_number]['RMSE']))
        
        bar = bars[index]
        if knn_matching == 'True':
            plt.bar(bar, rmse, color = colors[index], width = barWidthPlot, edgecolor='white', label = 'KNN matching')
        elif knn_matching == 'False':
            plt.bar(bar, rmse, color = colors[index], width = barWidthPlot, edgecolor='white', label = 'Lepard matching')
        index += 1

    plt.legend(loc='upper right')
    plt.xlabel("Model number")
    plt.ylabel("RMSE")
    title = data_type.replace('_', ' ')
    title = title.title()
    plt.title(title)
    plt.xticks([r + barWidth/2 for r in np.array([0, 1, 2, 3, 4, 5])], model_numbers)
    plt.savefig('Testing/all/compare_knn_lepard_' + data_type + '_' + feature_extractor + '_td_' + td + '_rmse.png')