
# initial tests using FCGF, KNN matching and Deformation Algorithm From Astrivis

training_data = 'full_deformed'
# training_data = 'partial_deformed'

epoch='10'
# epoch='5'

data_types = ['Full Non Deformed', 'Full Deformed', 'Partial Non Deformed', 'Partial Deformed']

final_data = {}

for data_type in data_types:
    data_type_mod = data_type.lower().replace(' ', '_')
    file_name = 'test_astrivis_' + data_type_mod + '_current_deformation_pre_none_fcgf_td_' + training_data + '_e_' + epoch + '_knn_True_conf_1e-06.txt'

    file = open(file_name, 'r')
    lines = file.readlines()
    list_keywords = ['full-epe', 'full-AccR', 'full-AccS', 'full-outlier', 'vis-epe', 'vis-AccS', 'vis-AccR', 'vis-outlier', 'RMSE', 'Relaxed IR', 'Strict IR']
    final_data = {}
    key = None

    for line in lines:
        if 'model' in line and len(line) < 35:
            if data:
                final_data[key] = data
            
            words = line.split(' ')
            model_number = words[1]