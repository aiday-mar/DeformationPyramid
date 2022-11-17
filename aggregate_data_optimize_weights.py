import re
import matplotlib.pyplot as plt

def data_file(file_path, deformed):

    file = open(file_path, 'r')
    lines = file.readlines()
    data = {}
    list_keywords = ['full-epe', 'full-AccR', 'full-AccS', 'full-outlier', 'vis-epe', 'vis-AccS', 'vis-AccR', 'vis-outlier', 'RMSE', 'IR']
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
                
    return final_data

def retrieve_type(obj, type, partial1 = None, partial2 = None):
    res = []
    for o in obj:
        if (partial1 and partial2 and obj[o]['partial1'] == partial1 and obj[o]['partial2'] == partial2) or (not partial1 and not partial2):
            for key in obj[o]:
                if key == type:
                    res.append(float(obj[o][key]))
    
    return res

def plot_all_for_one_type(data, title, number, partial1 = None, partial2 = None):
    f = plt.figure(number)
    plt.clf()
    RMSE = retrieve_type(data, 'RMSE', partial1, partial2)
    IR = retrieve_type(data, 'IR', partial1, partial2)
    full_epe = retrieve_type(data, 'full-epe', partial1, partial2)
    full_AccR = retrieve_type(data, 'full-AccR', partial1, partial2)
    full_AccS = retrieve_type(data, 'full-AccS', partial1, partial2)
    full_outlier = retrieve_type(data, 'full-outlier', partial1, partial2)
    vis_epe = retrieve_type(data, 'vis-epe', partial1, partial2)
    vis_AccR = retrieve_type(data, 'vis-AccR', partial1, partial2)
    vis_AccS = retrieve_type(data, 'vis-AccS', partial1, partial2)
    vis_outlier = retrieve_type(data, 'vis-outlier', partial1, partial2)
    plt.plot(RMSE)
    plt.plot(IR)
    plt.plot(full_epe)
    plt.plot(full_AccR)
    plt.plot(full_AccS)
    plt.plot(full_outlier)
    plt.plot(vis_epe)
    plt.plot(vis_AccR)
    plt.plot(vis_AccS)
    plt.plot(vis_outlier)
    plt.xlabel("Model from partial 0 to partial 1")
    plt.ylabel("Value")
    plt.legend(['RMSE', 'IR', 'full-epe', 'full-AccR', 'full-AccS', 'full-outlier', 'vis-epe', 'vis-AccR', 'vis-AccS', 'vis-outlier'])
    plt.title(title)

def plot_across_types(type, number, partial1, partial2):
    f = plt.figure(number)
    full_deformed = data_file('test_astrivis_full_deformed.txt', deformed =True)
    full_deformed = retrieve_type(full_deformed, type, partial1, partial2)
    partial_deformed = data_file('test_astrivis_partial_deformed.txt', deformed =True)
    partial_deformed = retrieve_type(partial_deformed, type, partial1, partial2)
    full_non_deformed = data_file('test_astrivis_full_non_deformed.txt', deformed =False)
    full_non_deformed = retrieve_type(full_non_deformed, type)
    partial_non_deformed = data_file('test_astrivis_partial_non_deformed.txt', deformed =False)
    partial_non_deformed = retrieve_type(partial_non_deformed, type)
    plt.plot(full_deformed)
    plt.plot(partial_deformed)
    plt.plot(full_non_deformed)
    plt.plot(partial_non_deformed)
    plt.xlabel("Model from partial " + partial1 + " to partial " + partial2)
    plt.ylabel(type)
    plt.legend(['full_deformed', 'partial deformed', 'full non deformed', 'partial non deformed'])
    plt.title(type)

# When the type is fixed
data_full_deformed = data_file('test_astrivis_full_deformed.txt', deformed =True)
plot_all_for_one_type(data_full_deformed, 'Full Deformed', 1, '0', '1')

data_full_non_deformed = data_file('test_astrivis_full_non_deformed.txt', deformed = False)
plot_all_for_one_type(data_full_non_deformed, 'Full Non Deformed', 2)

data_partial_deformed = data_file('test_astrivis_partial_deformed.txt', deformed =True)
plot_all_for_one_type(data_partial_deformed, 'Partial Deformed', 3, '0', '1')

data_partial_non_deformed = data_file('test_astrivis_partial_non_deformed.txt', deformed = False)
plot_all_for_one_type(data_partial_non_deformed, 'Partial Non Deformed', 4)

# When the measure is fixed
plot_across_types('RMSE', 5, '0', '1')
plot_across_types('IR', 6, '0', '1')
plot_across_types('full-epe', 7, '0', '1')
plot_across_types('full-AccR', 8, '0', '1')
plot_across_types('full-AccS', 9, '0', '1')
plot_across_types('full-outlier', 10, '0', '1')
plot_across_types('vis-epe', 11, '0', '1')
plot_across_types('vis-AccR', 12, '0', '1')
plot_across_types('vis-AccS', 13, '0', '1')
plot_across_types('vis-outlier', 14, '0', '1')
plt.show()