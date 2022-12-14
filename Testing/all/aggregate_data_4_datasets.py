import re
import matplotlib.pyplot as plt

# FCGF DONE
# KPFCN TODO

feature_extractor='fcgf'
# feature_extractor='kpfcn'

# preprocessing='none'
preprocessing='mutual'

# training_data='full_deformed'
training_data='partial_deformed'

epoch='1'
# epoch='2'
# epoch='none'

# custom=True
custom=False
custom_val = '_custom' if custom else ''
custom_val_title = ' - custom' if custom else ''
adm_val = '_adm_3.0' if custom else ''
adm_val_title = '3.0' if custom else ''

if training_data == 'full_deformed':
    training_data_val = 'full deformed'
elif training_data == 'partial_deformed':
    training_data_val = 'partial deformed'
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

def plot_all_for_one_type(data, title, number, partial1 = None, partial2 = None, save_path=None):
    plt.clf()
    f = plt.figure(number)
    RMSE = retrieve_type(data, 'RMSE', partial1, partial2)
    strict_IR = retrieve_type(data, 'Strict IR', partial1, partial2)
    relaxed_IR = retrieve_type(data, 'Relaxed IR', partial1, partial2)
    full_epe = retrieve_type(data, 'full-epe', partial1, partial2)
    full_AccR = retrieve_type(data, 'full-AccR', partial1, partial2)
    full_AccS = retrieve_type(data, 'full-AccS', partial1, partial2)
    full_outlier = retrieve_type(data, 'full-outlier', partial1, partial2)
    vis_epe = retrieve_type(data, 'vis-epe', partial1, partial2)
    vis_AccR = retrieve_type(data, 'vis-AccR', partial1, partial2)
    vis_AccS = retrieve_type(data, 'vis-AccS', partial1, partial2)
    vis_outlier = retrieve_type(data, 'vis-outlier', partial1, partial2)
    plt.plot(model_numbers, RMSE)
    plt.plot(model_numbers, strict_IR)
    plt.plot(model_numbers, relaxed_IR)
    plt.plot(model_numbers, full_epe)
    plt.plot(model_numbers, full_AccR)
    plt.plot(model_numbers, full_AccS)
    plt.plot(model_numbers, full_outlier)
    plt.plot(model_numbers, vis_epe)
    plt.plot(model_numbers, vis_AccR)
    plt.plot(model_numbers, vis_AccS)
    plt.plot(model_numbers, vis_outlier)
    plt.xlabel("Model number")
    plt.ylabel("Value")
    plt.legend(['RMSE', 'Strict IR', 'Relaxed IR', 'full-epe', 'full-AccR', 'full-AccS', 'full-outlier', 'vis-epe', 'vis-AccR', 'vis-AccS', 'vis-outlier'])
    if feature_extractor == 'fcgf':
        plt.title(title + ' - ' + 'FCGF feature extractor' + ' - ' + ' trained on ' +  training_data_val + ' data' + ' - ' + ' epoch ' + epoch + custom_val_title + ' ' + adm_val_title, wrap=True)
    elif feature_extractor == 'kpfcn':
        plt.title(title + ' - ' + 'KPFCN feature extractor' + ' - ' + ' trained on ' +  training_data_val + ' data' + ' - ' + ' epoch ' + epoch + custom_val_title + ' ' + adm_val_title, wrap=True)
    if save_path:
        plt.savefig(save_path)

def plot_across_types(type, number, partial1, partial2, save_path = None):
    plt.clf()
    f = plt.figure(number)
    full_deformed = data_file('Testing/all/test_astrivis_full_deformed_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.txt', deformed =True)
    full_deformed = retrieve_type(full_deformed, type, partial1, partial2)
    full_non_deformed = data_file('Testing/all/test_astrivis_full_non_deformed_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.txt', deformed =False)
    full_non_deformed = retrieve_type(full_non_deformed, type)
    partial_deformed = data_file('Testing/all/test_astrivis_partial_deformed_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val  + adm_val + '.txt', deformed =True)
    partial_deformed = retrieve_type(partial_deformed, type, partial1, partial2)
    partial_non_deformed = data_file('Testing/all/test_astrivis_partial_non_deformed_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.txt', deformed =False)
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

# When the type is fixed
data_full_deformed = data_file('Testing/all/test_astrivis_full_deformed_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.txt', deformed =True)
plot_all_for_one_type(data_full_deformed, 'Full Deformed - all metrics', 1, partial_scan_1, partial_scan_2, save_path='Testing/all/full_deformed_all_metrics_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val +  adm_val + '.png')

data_full_non_deformed = data_file('Testing/all/test_astrivis_full_non_deformed_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.txt', deformed = False)
plot_all_for_one_type(data_full_non_deformed, 'Full Non Deformed - all metrics', 2, save_path='Testing/all/full_non_deformed_all_metrics_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')

data_partial_deformed = data_file('Testing/all/test_astrivis_partial_deformed_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.txt', deformed =True)
plot_all_for_one_type(data_partial_deformed, 'Partial Deformed - all metrics', 3, partial_scan_1, partial_scan_2, save_path='Testing/all/partial_deformed_all_metrics_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')

data_partial_non_deformed = data_file('Testing/all/test_astrivis_partial_non_deformed_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.txt', deformed = False)
plot_all_for_one_type(data_partial_non_deformed, 'Partial Non Deformed - all metrics', 4, save_path='Testing/all/partial_non_deformed_all_metrics_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')

# When the measure is fixed
plot_across_types('RMSE', 5, partial_scan_1, partial_scan_2, save_path='Testing/all/all_data_types_rmse_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
plot_across_types('Strict IR', 6, partial_scan_1, partial_scan_2, save_path='Testing/all/all_data_types_strict_ir_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
plot_across_types('Relaxed IR', 6, partial_scan_1, partial_scan_2, save_path='Testing/all/all_data_types_relaxed_ir_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
plot_across_types('full-epe', 7, partial_scan_1, partial_scan_2, save_path='Testing/all/all_data_types_full_epe_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
plot_across_types('full-AccR', 8, partial_scan_1, partial_scan_2, save_path='Testing/all/all_data_types_full_accr_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val +  adm_val + '.png')
plot_across_types('full-AccS', 9, partial_scan_1, partial_scan_2, save_path='Testing/all/all_data_types_full_accs_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
plot_across_types('full-outlier', 10, partial_scan_1, partial_scan_2, save_path='Testing/all/all_data_types_full_outlier_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
plot_across_types('vis-epe', 11, partial_scan_1, partial_scan_2, save_path='Testing/all/all_data_types_vis_epe_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
plot_across_types('vis-AccR', 12, partial_scan_1, partial_scan_2, save_path='Testing/all/all_data_types_vis_accr_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
plot_across_types('vis-AccS', 13, partial_scan_1, partial_scan_2, save_path='Testing/all/all_data_types_vis_accs_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
plot_across_types('vis-outlier', 14, partial_scan_1, partial_scan_2, save_path='Testing/all/all_data_types_vis_outlier_' + feature_extractor + '_td_' + training_data + '_e_' + epoch + custom_val + adm_val + '.png')
