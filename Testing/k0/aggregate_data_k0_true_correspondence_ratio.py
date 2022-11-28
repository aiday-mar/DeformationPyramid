import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class File:
    def __init__(self, type, preprocessing, confidence, number_centers, average_distance_multiplier, coarse_level, index_coarse, number_iterations, inlier_outlier_thr, sampling):
        self.type = type
        self.preprocessing = preprocessing
        self.confidence = confidence
        self.number_centers = number_centers
        self.average_distance_multiplier = average_distance_multiplier
        self.coarse_level = coarse_level
        self.index_coarse = index_coarse
        self.number_iterations = number_iterations
        self.inlier_outlier_thr = inlier_outlier_thr
        self.sampling = sampling
    
    def __str__(self):
        return " - Type : " + str(self.type) + " - Preprocessing : " + str(self.preprocessing) + " - Confidence : " + str(self.confidence) + " - Number Centers : " + str(self.number_centers) + " - Average Distance Multiplier : " + str(self.average_distance_multiplier) + " - Coarse Level : " + str(self.coarse_level) + " - Index Coarse : " + str(self.index_coarse) + ' - Number Iterations : ' + str(self.number_iterations) + ' - Inlier Outlier Threshold : ' + str(self.inlier_outlier_thr) + ' - Sampling : ' + str(self.sampling)

files=[]
file_types=[]
number_iterations=1
data_types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
base = 'Testing/'
folder = 'k0/'
file='testing_k0.txt'
title = 'Varying k0'

k0 = [-11, -10, -9, -8, -7, -6, -5]
shape = (len(k0),)

final_matrices={'Full Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}, 
                'Full Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}, 
                'Partial Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}},  
                'Partial Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}}

file_txt = open(base + folder + file, 'r')
Lines = file_txt.readlines()
k0_val = -1
current_data_type = ''
for line in Lines:
    if line[:-1] in data_types:
        current_data_type = line[:-1]
    if 'Test - k0' in line:
        k0_val = int(re.findall(r'\d+', line)[1])
        print(k0_val)
    if 'number of true landmarks correspondences returned from Lepard' in line:
        search = list(map(int, re.findall(r'\d+', line)))
        true = int(search[0])
        total = int(search[1])
        i = k0.index(k0_val)
        final_matrices[current_data_type]['lepard']['true'][i] = true
        final_matrices[current_data_type]['lepard']['total'][i] = total - true
        

for data_type in data_types:
    plt.clf()
    true_data = []
    total_data = []
    fraction = []
    
    for i in range(len(k0)):
                
        true_data.append(final_matrices[data_type]['lepard']['true'][i])
        total_data.append(final_matrices[data_type]['lepard']['total'][i])
          
        if final_matrices[data_type]['lepard']['total'][i] != 0:
            fraction.append(final_matrices[data_type]['lepard']['true'][i]/(final_matrices[data_type]['lepard']['total'][i]+final_matrices[data_type]['lepard']['true'][i]))
        else:
            fraction.append(0)

    plt.title(title)
    k0_pos = range(0, len(k0))
    plt.bar(k0_pos, true_data, color='r')
    plt.bar(k0_pos, total_data, bottom=true_data, color='b')
    plt.xticks(k0_pos, k0, rotation=90)
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_bar_chart.png', bbox_inches='tight')
    
    plt.clf()
    plt.title(title)
    plt.plot(k0_pos, fraction, color='r')
    plt.xticks(k0_pos, k0, rotation=90)
    plt.ylim(0, 1)
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_graph.png', bbox_inches='tight')