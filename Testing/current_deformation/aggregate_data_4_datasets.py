import re
import matplotlib.pyplot as plt

# FCGF DONE
# KPFCN TODO

# feature_extractor='fcgf'
feature_extractor='kpfcn'

preprocessing='none'
# preprocessing='mutual'

# training_data='full_deformed'
# training_data='partial_deformed'
training_data='pretrained'

# epoch='1'
# epoch='2'
epoch='null'

custom=False
# custom=True

custom_val = '_custom' if custom else ''
custom_val_title = ' - custom' if custom else ''
adm_val = '_adm_2.0' if custom else ''
adm_val_title = '2.0' if custom else ''

if training_data == 'full_deformed':
    training_data_val = 'full deformed'
elif training_data == 'partial_deformed':
    training_data_val = 'partial deformed'
elif training_data == 'pretrained':
    training_data_val = 'pretrained'    
else:
    raise Exception('specify a valid training dataset')

partial_scan_1 = '020'
partial_scan_2 = '104'

model_numbers = ['002', '042', '085', '126', '167', '207']

def data_file(file_path, deformed):

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

def retrieve_type(obj, type, partial1 = None, partial2 = None):
    res = []
    for o in obj:
        if (partial1 and partial2 and obj[o]['partial1'] == partial1 and obj[o]['partial2'] == partial2) or (not partial1 and not partial2):
            for key in obj[o]:
                if key == type:
                    res.append(float(obj[o][key]))
    
    return res

def plot_across_types(type, number, partial1, partial2, save_path = None):
    plt.clf()
    f = plt.figure(number)
    full_deformed = data_file("Testing/current_deformation/test_astrivis_full_deformed_pre_current_deformation_pre_" + preprocessing_normal + "_" + feature_extractor + "_td_" + training_data + "_e_" + epoch + custom_val + adm_val + '.txt', deformed =True)
    full_deformed = retrieve_type(full_deformed, type, partial1, partial2)
    full_non_deformed = data_file("Testing/current_deformation/test_astrivis_full_non_deformed_pre_current_deformation_pre_" + preprocessing_normal + "_" + feature_extractor + "_td_" + training_data + "_e_" + epoch + custom_val + adm_val + '.txt', deformed =False)
    full_non_deformed = retrieve_type(full_non_deformed, type)
    partial_deformed = data_file("Testing/current_deformation/test_astrivis_partial_deformed_pre_current_deformation_pre_" + preprocessing_normal + "_" + feature_extractor + "_td_" + training_data + "_e_" + epoch + custom_val + adm_val + '.txt', deformed =True)
    partial_deformed = retrieve_type(partial_deformed, type, partial1, partial2)
    partial_non_deformed = data_file("Testing/current_deformation/test_astrivis_partial_non_deformed_pre_current_deformation_pre_" + preprocessing_normal + "_" + feature_extractor + "_td_" + training_data + "_e_" + epoch + custom_val + adm_val + '.txt', deformed =False)
    partial_non_deformed = retrieve_type(partial_non_deformed, type)
    plt.plot(model_numbers, full_deformed)
    plt.plot(model_numbers, partial_deformed)
    plt.plot(model_numbers, full_non_deformed)
    plt.plot(model_numbers, partial_non_deformed)
    plt.xlabel("Model number")
    plt.ylabel(type)
    plt.legend(['Full Deformed', 'Partial Deformed', 'Full Non Deformed', 'Partial Non Deformed'])
    if feature_extractor == 'fcgf':
        plt.title(type + ' - ' + 'FCGF feature extractor'  + ' - ' + ' trained on ' +  training_data_val + ' data' + ' - ' + ' epoch ' + epoch + custom_val_title + ' ' + adm_val_title, wrap=True)
    elif feature_extractor == 'kpfcn':
        plt.title(type + ' - ' + 'KPFCN feature extractor'  + ' - ' + ' trained on ' +  training_data_val + ' data' + ' - ' + ' epoch ' + epoch + custom_val_title + ' ' + adm_val_title, wrap=True)
    
    if save_path:
        plt.savefig(save_path)

plot_across_types('RMSE', 5, partial_scan_1, partial_scan_2, save_path='Testing/current_deformation/all_data_types_rmse_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
plot_across_types('Strict IR', 6, partial_scan_1, partial_scan_2, save_path='Testing/current_deformation/all_data_types_strict_ir_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
plot_across_types('Relaxed IR', 6, partial_scan_1, partial_scan_2, save_path='Testing/current_deformation/all_data_types_relaxed_ir_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')