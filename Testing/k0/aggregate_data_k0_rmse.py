import re
import matplotlib.pyplot as plt
import numpy as np

data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'Testing/'
folder = 'k0/'
file='testing_k0.txt'
title = 'RMSE - Varying k0'

k0_list=[-11 -10 -9 -8]
shape = (len(k0_list),)

final_matrices={'Full Non Deformed': {'rmse' : np.zeros(shape)}, 
                'Full Deformed': {'rmse' : np.zeros(shape)}, 
                'Partial Deformed': {'rmse' : np.zeros(shape)},  
                'Partial Non Deformed': {'rmse' : np.zeros(shape)}}

file_txt = open(base + folder + file, 'r')
Lines = file_txt.readlines()
k0_val = -1
current_data_type = ''
for line in Lines:
    if line[:-1] in data_types:
        current_data_type = line[:-1]
    if 'Test - k0' in line:
        k0_val = int(re.findall('-?\d+', line)[1])
    if 'RMSE' in line:
        rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = k0_list.index(k0_val)
        final_matrices[current_data_type]['rmse'][i] = rmse
        
print('final_matrices : ', final_matrices)

for data_type in data_types:
    plt.clf()
    k0_pos = range(len(k0_list))
    plt.clf()
    plt.title(title)
    plt.plot(k0_pos, final_matrices[data_type]['rmse'], color='r')
    plt.xticks(k0_pos, k0_list, rotation=90)
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_graph.png', bbox_inches='tight')