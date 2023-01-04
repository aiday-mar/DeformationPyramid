
import re
import matplotlib.pyplot as plt

data_types = ['full_deformed', 'partial_deformed', 'full_non_deformed', 'partial_non_deformed']

weights = {
    'fcgf' : {
        # 'full_deformed' : 10, 
        'partial_deformed' : 5
    }, 
    'kpfcn' : {
        'full_deformed' : 10, 
        'partial_deformed' : 5,
        'pretrained' : 'null'
    }
}

number = 0
model_numbers = ['002', '042', '085', '126', '167', '207']
adm = 2.0
# with_custom = False
with_custom = True
colors = ['blue', 'orange', 'green', 'red', 'magenta']

preprocessing_custom='none'
# preprocessing_custom='mutual'

preprocessing_normal='mutual'
# preprocessing_normal='none'

def get_data(data_type, feature_extractor, training_data_type, custom = False):
    if data_type == 'full_deformed' or data_type == 'partial_deformed':
        deformed = True
    else:
        deformed = False

    epoch = str(weights[feature_extractor][training_data_type])
    if custom is False:
        file_path = 'Testing/all/test_astrivis_' + data_type + '_pre_' + preprocessing_normal + '_' + feature_extractor + '_td_' + training_data_type + '_e_' + epoch + '.txt'
    else:
        file_path = 'Testing/all/test_astrivis_' + data_type + '_pre_' + preprocessing_custom + '_' + feature_extractor + '_td_' + training_data_type + '_e_' + epoch + '_custom_adm_' + str(adm) + '.txt'
    
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
    legend = []

    if with_custom is False:
        color_idx = 0
        for feature_extractor in weights:
            for training_data_type in weights[feature_extractor]:

                epoch = str(weights[feature_extractor][training_data_type])
                training_data_type_mod = training_data_type.replace('_', ' ')
                weights_legend = feature_extractor + ' - ' + training_data_type_mod + ' - ' + epoch 
                legend.append(weights_legend)
                data = get_data(data_type, feature_extractor, training_data_type)
                rmse = []             
                for model_number in data:
                    rmse.append(float(data[model_number]['RMSE']))
                
                plt.plot(model_numbers, rmse, color = colors[color_idx], label=weights_legend)
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
    plt.legend(loc='upper right') # legend
    title = data_type.replace('_', ' ')
    title = title.title()
    plt.title(title, wrap=True)
    if with_custom is False:
        plt.savefig('Testing/all/per_data_type_' + data_type + '_pre_' + preprocessing_normal + '_rmse.png')
    else:
        plt.savefig('Testing/all/per_data_type_' + data_type + '_pre_' + preprocessing_custom + ' _rmse_custom.png')

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