
import re
import matplotlib.pyplot as plt

data_types = ['full_deformed', 'partial_deformed', 'full_non_deformed', 'partial_non_deformed']

weights = {
    'fcgf' : {
        'full_deformed' : 2, 
        'partial_deformed' : 1
    }, 
    'kpfcn' : {
        'full_deformed' : 2, 
        'pretrained' : 'null'
    }
}

number = 0
model_numbers = ['002', '042', '085', '126', '167', '207']

def get_data(data_type, feature_extractor, training_data_type):
    if data_type == 'full_deformed' or data_type == 'partial_deformed':
        deformed = True
    else:
        deformed = False

    # print('weights[feature_extractor] : ', weights[feature_extractor])
    epoch = str(weights[feature_extractor][training_data_type])
    file_path = 'Testing/all/test_astrivis_' + data_type + '_pre_mutual_' + feature_extractor + '_td_' + training_data_type + '_e_' + epoch + '.txt'
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

    for feature_extractor in weights:
        for training_data_type in weights[feature_extractor]:
            
            epoch = str(weights[feature_extractor][training_data_type])
            weights_legend = feature_extractor + ' - ' + training_data_type + ' - ' + epoch 
            legend.append(weights_legend)
            data = get_data(data_type, feature_extractor, training_data_type)
            rmse = []             
            for model_number in data:
                rmse.append(float(data[model_number]['RMSE']))
            
            plt.plot(model_numbers, rmse)
    
    plt.xlabel("Model number")
    plt.ylabel("RMSE")
    plt.legend(legend)
    plt.title(data_type, wrap=True)
    plt.savefig('Testing/all/per_data_type_' + data_type + '_rmse.png')
            