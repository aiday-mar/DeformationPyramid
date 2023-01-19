import re
import matplotlib.pyplot as plt
import numpy as np
import copy

data_types=['Full Deformed', 'Partial Deformed']
base = 'Testing/custom_filtering/'

model_numbers = ['002', '042', '085', '126', '167', '207']
# preprocessing = 'none'
preprocessing = 'mutual'
max_ldmks = 'None'

adm_changed=True
# adm_changed=False

if preprocessing == 'mutual':
    if adm_changed is False:
        weights = {
            'kpfcn' : {
                'full_deformed' : {
                    'conf' : '0.000001',
                    'nc' : [5, 10, 15]
                },
                'partial_deformed' : {
                    'conf' : '0.000001',
                    'nc' : [5, 10, 15]
                }
            },
            'fcgf' : {
                'full_deformed' : {
                    'conf' : '0.000001',
                    'nc' : [10, 50, 100, 150, 200]
                },
                'partial_deformed' : {
                    'conf' : '0.000001',
                    'nc' : [10, 50, 100, 150]
                }
            }   
        }
    else:
        weights = {
            # 'kpfcn' : {
            #    'full_deformed' : {
            #        'conf' : '0.000001',
            #        'adm' : [1, 2, 3, 4, 5, 6]
            #    },
            #    'partial_deformed' : {
            #        'conf' : '0.000001',
            #        'adm' : [1, 2, 3, 4, 5, 6]
            #    }
            # },
            'fcgf' : {
                'full_deformed' : {
                    'conf' : '0.000001',
                    'adm' : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                },
            #    'partial_deformed' : {
            #        'conf' : '0.000001',
            #        'adm' : [1, 2, 3, 4, 5, 6]
            #    }
            }   
        }

elif preprocessing == 'none':
    if adm_changed is False:
        weights = {
            # 'kpfcn' : {
            #    'full_deformed' : '0.000001'
            #    'partial_deformed' : '0.000001'
            # }
            'fcgf' : {
                # 'full_deformed' : {
                #    'conf' : '0.000001',
                #    'nc' : [10, 50, 100, 150, 200]
                # },
                'partial_deformed' : {
                    'conf' : '0.000001',
                    'nc' : [50, 100, 200, 300, 500, 700]
                }
            }   
        }
    else:
        weights = {}
else:
    raise Exception('Must be one of the preprocessing options')

confidence = '1e-06'
# adm = [1.0, 2.0, 3.0, 4.0, 5.0]
adm = [3.0]
nc = [150]
iot = [0.01]

# sampling = 'poisson'
sampling='linspace'

def create_final_matrix(var, type):

    if type == 'adm':
        adm = var
        nc = [150]
    elif type == 'nc':
        nc = var
        adm = [3.0]

    shape=(len(nc), len(adm), len(iot))

    final_submatrix ={
    'Full Deformed': {'lepard' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'outlier' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}, 'n_distinct' : 0}, 
    'Partial Deformed': {'lepard' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'outlier' : {'total' : 0, 'true' : 0, 'rmse': 0}, 'custom' : {'total' : np.zeros(shape), 'true' : np.zeros(shape), 'rmse': np.zeros(shape)}, 'n_distinct' : 0},  
    }

    final_matrix = { model_number : copy.deepcopy(final_submatrix) for model_number in model_numbers}
    return final_matrix

final_matrices = {}

for feature_extractor in weights:
    final_matrices[feature_extractor] = {}
    for training_data in weights[feature_extractor]:
        if adm_changed is True:
            final_matrix = create_final_matrix(weights[feature_extractor][training_data]['adm'], 'adm')
        else:
            final_matrix = create_final_matrix(weights[feature_extractor][training_data]['nc'], 'nc')
        final_matrices[feature_extractor][training_data] = copy.deepcopy(final_matrix)

for feature_extractor in weights:
    for training_data in weights[feature_extractor]:

        confidence = weights[feature_extractor][training_data]['conf']
        if adm_changed is True:
            adm = weights[feature_extractor][training_data]['adm']
        else:
            nc = weights[feature_extractor][training_data]['nc']

        for model_number in model_numbers:
            count = 0
            for i in nc :
                for j in adm:
                    for k in iot:
                        file = 'p_' + preprocessing + '_c_' + confidence + '_nc_' + str(i) + '_adm_' + str(j) + '_iot_' + str(k) + '_s_' + sampling + '_max_ldmks_' +  max_ldmks + '_' + feature_extractor + '_td_' + training_data + '_model_' + model_number + '.txt'
                        file_txt = open(base + file, 'r')
                        Lines = file_txt.readlines()
                        current_data_type = ''
                        for line in Lines:
                            if line[:-1] in data_types:
                                current_data_type = line[:-1]
                                    
                            if 'number of true landmark correspondences returned from custom filtering' in line:
                                search = list(map(int, re.findall(r'\d+', line)))
                                true = int(search[0])
                                total = int(search[1])
                                if adm_changed is True:
                                    final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['true'][0][count][0] = true
                                    final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['total'][0][count][0] = total - true
                                else:
                                    final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['true'][count][0][0] = true
                                    final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['total'][count][0][0] = total - true
                            
                            if 'RMSE' in line:
                                rmse = float(re.findall("\d+\.\d+", line)[0])
                                if adm_changed is True:
                                    final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['rmse'][0][count][0] = rmse
                                else:
                                    final_matrices[feature_extractor][training_data][model_number][current_data_type]['custom']['rmse'][count][0][0] = rmse
                                
                            if 'number of distinct source landmarks ' in line:
                                search = list(map(int, re.findall(r'\d+', line)))
                                n_distinct = search[0]
                                final_matrices[feature_extractor][training_data][model_number][current_data_type]['n_distinct'] = n_distinct

                    if adm_changed is True:      
                        count += 1

                if adm_changed is False:      
                    count += 1

for feature_extractor in weights:
    for training_data in weights[feature_extractor]:
        for model_number in model_numbers:
            file_txt = 'Testing/custom_filtering/output_lepard_default_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_model_' + model_number + '.txt'
            file_txt = open(file_txt, 'r')
            Lines = file_txt.readlines()
            current_data_type = ''
            for line in Lines:
                if line[:-1] in data_types:
                    current_data_type = line[:-1]

                line_to_check = ''

                if feature_extractor == 'kpfcn':
                    line_to_check = 'number of true landmarks correspondences returned from Lepard'
                elif feature_extractor == 'fcgf':
                    line_to_check = 'number of true landmarks correspondences returned from FCGF based Lepard'

                if line_to_check in line:
                    search = list(map(int, re.findall(r'\d+', line)))
                    true = int(search[0])
                    total = int(search[1])
                    if adm_changed is True:
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['lepard']['true'] = true
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['lepard']['total'] = total - true
                    else:
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['lepard']['true'] = true
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['lepard']['total'] = total - true

                if 'RMSE' in line:
                    rmse = float(re.findall("\d+\.\d+", line)[0])
                    if adm_changed is True:
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['lepard']['rmse'] = rmse
                    else:
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['lepard']['rmse'] = rmse

for feature_extractor in weights:
    for training_data in weights[feature_extractor]:
        for model_number in model_numbers:
            file_txt = 'Testing/custom_filtering/output_outlier_rejection_default_pre_' + preprocessing + '_' + feature_extractor + '_td_' + training_data + '_model_' + model_number + '.txt'
            file_txt = open(file_txt, 'r')
            Lines = file_txt.readlines()
            current_data_type = ''
            count = 0
            for line in Lines:
                if line[:-1] in data_types:
                    current_data_type = line[:-1]

                line_to_check = ''

                if feature_extractor == 'kpfcn':
                    line_to_check = 'number of true landmark correspondences returned from Outlier Rejection'
                elif feature_extractor == 'fcgf':
                    line_to_check = 'number of true landmark correspondences returned from FCGF based Outlier Rejection'

                if line_to_check in line:
                    search = list(map(int, re.findall(r'\d+', line)))
                    true = int(search[0])
                    total = int(search[1])
                    if adm_changed is True:
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['outlier']['true'] = true
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['outlier']['total'] = total - true
                    else:
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['outlier']['true'] = true
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['outlier']['total'] = total - true
                
                if 'RMSE' in line:
                    rmse = float(re.findall("\d+\.\d+", line)[0])
                    if adm_changed is True:
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['outlier']['rmse'] = rmse
                    else:
                        final_matrices[feature_extractor][training_data][model_number][current_data_type]['outlier']['rmse'] = rmse

 # Changing ADM
'''
modified_adm = ['custom - ' + str(adm_r) for adm_r in adm]
modified_adm_pos = range(len(modified_adm))

plt.title(data_type + ' - varying ADM')
plt.plot(modified_adm_pos, rmse, color='r')
plt.xticks(modified_adm_pos, modified_adm, rotation=90)
plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_rmse_nc_' + str(nc[0]) + '_iot_' + str(iot[0]) + '_sampling_' + sampling + '_varying_adm.png', bbox_inches='tight')

modified_adm_lepard_outlier = ['custom - ' + str(adm_r) for adm_r in adm]
modified_adm_lepard_outlier.append('lepard')
modified_adm_lepard_outlier.append('outlier rejection')
modified_adm_lepard_outlier_pos = range(len(modified_adm_lepard_outlier))

plt.title(data_type + ' - varying ADM')
plt.bar(modified_adm_lepard_outlier_pos, true_data, color='r')
plt.bar(modified_adm_lepard_outlier_pos, total_data, bottom=true_data, color='b')
plt.xticks(modified_adm_lepard_outlier_pos, modified_adm_lepard_outlier, rotation=90)
plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_gt_ratio_barchart_nc_' + str(nc[0]) + '_iot_' + str(iot[0]) + '_sampling_' + sampling + '_varying_adm.png', bbox_inches='tight')

plt.clf()
plt.title(data_type + ' - varying ADM')
plt.plot(modified_adm_lepard_outlier_pos, fraction, color='r')
plt.xticks(modified_adm_lepard_outlier_pos, modified_adm_lepard_outlier, rotation=90)
plt.ylim(0, 1)
plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_gt_ratio_graph_nc_' + str(nc[0]) + '_iot_' + str(iot[0]) + '_sampling_' + sampling + '_varying_adm.png', bbox_inches='tight')
'''

print('final_matrices : ', final_matrices)

if adm_changed is False:
    for data_type in data_types:
        
        plt.clf()

        for feature_extractor in weights:
            for training_data in weights[feature_extractor]:
                
                nc = weights[feature_extractor][training_data]['nc']
                modified_nc = [str(nc_v) for nc_v in nc]
                modified_nc.append('lepard')
                modified_nc.append('outlier rejection')
                modified_nc_pos = range(len(modified_nc))

                if 'Full' in data_type and 'partial' in training_data or 'Partial' in data_type and 'full' in training_data:
                    continue

                for model_number in model_numbers:
            
                    true_data = []
                    total_data = []
                    
                    for i in range(len(nc)) :
                        for j in range(len(adm)):
                            for k in range(len(iot)):
                                    
                                true_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['custom']['true'][i][j][k])
                                total_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['custom']['total'][i][j][k])
                        
                    true_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['lepard']['true'])
                    total_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['lepard']['total'])

                    true_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['outlier']['true'])
                    total_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['outlier']['total'])

                    plt.clf()
                    plt.title('Model ' + model_number + ' - ' + data_type)
                    plt.bar(modified_nc_pos, true_data, color='r')
                    plt.bar(modified_nc_pos, total_data, bottom=true_data, color='b')
                    plt.xticks(modified_nc_pos, modified_nc, rotation=90)
                    plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_pre_' + preprocessing + '_max_ldmks_' + max_ldmks + '_c_' + confidence + '_adm_' + str(adm[0]) + '_iot_' + str(iot[0]) + '_s_' + sampling + '_' + feature_extractor + '_td_' + training_data + '_varying_nc_gt_ratio_model_' + model_number + '.png', bbox_inches='tight')
                
        
        for feature_extractor in weights:
            for training_data in weights[feature_extractor]:
                
                plt.clf()
                nc = weights[feature_extractor][training_data]['nc']
                modified_nc = [str(nc_v) for nc_v in nc]
                modified_nc.append('lepard')
                modified_nc.append('outlier rejection')
                modified_nc_pos = range(len(modified_nc))

                if 'Full' in data_type and 'partial' in training_data or 'Partial' in data_type and 'full' in training_data:
                    continue
                
                for model_number in model_numbers:
            
                    rmse = []
                    
                    for i in range(len(nc)) :
                        for j in range(len(adm)):
                            for k in range(len(iot)):

                                rmse.append(final_matrices[feature_extractor][training_data][model_number][data_type]['custom']['rmse'][i][j][k])
                        
                    rmse.append(final_matrices[feature_extractor][training_data][model_number][data_type]['lepard']['rmse'])
                    rmse.append(final_matrices[feature_extractor][training_data][model_number][data_type]['outlier']['rmse'])

                    plt.plot(modified_nc_pos, rmse, label = model_number)

                plt.title(data_type)
                plt.legend(loc ="upper right")
                plt.xticks(modified_nc_pos, modified_nc, rotation=90)
                plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_pre_' + preprocessing + '_max_ldmks_' + max_ldmks + '_c_' + confidence + '_adm_' + str(adm[0]) + '_iot_' + str(iot[0]) + '_s_' + sampling + '_' + feature_extractor + '_td_' + training_data + '_varying_nc_rmse.png', bbox_inches='tight')
else:

    for data_type in data_types:
        
        plt.clf()
        for feature_extractor in weights:
            for training_data in weights[feature_extractor]:
                
                adm = weights[feature_extractor][training_data]['adm']
                modified_adm = [str(adm_v) for adm_v in adm]
                modified_adm.append('lepard')
                modified_adm.append('outlier rejection')
                modified_adm_pos = range(len(modified_adm))

                if 'Full' in data_type and 'partial' in training_data or 'Partial' in data_type and 'full' in training_data:
                    continue

                for model_number in model_numbers:
            
                    true_data = []
                    total_data = []
                    
                    for i in range(len(nc)) :
                        for j in range(len(adm)):
                            for k in range(len(iot)):
                                    
                                true_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['custom']['true'][i][j][k])
                                total_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['custom']['total'][i][j][k])
                        
                    true_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['lepard']['true'])
                    total_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['lepard']['total'])

                    true_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['outlier']['true'])
                    total_data.append(final_matrices[feature_extractor][training_data][model_number][data_type]['outlier']['total'])

                    plt.clf()
                    plt.title('Model ' + model_number + ' - ' + data_type)
                    plt.bar(modified_adm_pos, true_data, color='r')
                    plt.bar(modified_adm_pos, total_data, bottom=true_data, color='b')
                    plt.xticks(modified_adm_pos, modified_adm, rotation=90)
                    plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_pre_' + preprocessing + '_max_ldmks_' + max_ldmks + '_c_' + confidence + '_nc_' + str(nc[0]) + '_iot_' + str(iot[0]) + '_s_' + sampling + '_' + feature_extractor + '_td_' + training_data + '_varying_adm_gt_ratio_model_' + model_number + '.png', bbox_inches='tight')
                
        
        for feature_extractor in weights:
            for training_data in weights[feature_extractor]:
                
                plt.clf()
                adm = weights[feature_extractor][training_data]['adm']
                modified_adm = [str(adm_v) for adm_v in adm]
                modified_adm.append('lepard')
                modified_adm.append('outlier rejection')
                modified_adm_pos = range(len(modified_adm))

                if 'Full' in data_type and 'partial' in training_data or 'Partial' in data_type and 'full' in training_data:
                    continue
                
                for model_number in model_numbers:
            
                    rmse = []
                    
                    for i in range(len(nc)) :
                        for j in range(len(adm)):
                            for k in range(len(iot)):

                                rmse.append(final_matrices[feature_extractor][training_data][model_number][data_type]['custom']['rmse'][i][j][k])
                        
                    rmse.append(final_matrices[feature_extractor][training_data][model_number][data_type]['lepard']['rmse'])
                    rmse.append(final_matrices[feature_extractor][training_data][model_number][data_type]['outlier']['rmse'])

                    plt.plot(modified_adm_pos, rmse, label = model_number)

                plt.title(data_type)
                plt.legend(loc ="upper right")
                plt.xticks(modified_adm_pos, modified_adm, rotation=90)
                plt.savefig('Testing/custom_filtering/' + data_type.replace(' ', '_') + '_pre_' + preprocessing + '_max_ldmks_' + max_ldmks + '_c_' + confidence + '_nc_' + str(nc[0]) + '_iot_' + str(iot[0]) + '_s_' + sampling + '_' + feature_extractor + '_td_' + training_data + '_varying_adm_rmse.png', bbox_inches='tight')
                        