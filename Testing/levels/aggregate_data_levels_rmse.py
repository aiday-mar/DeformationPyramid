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
folder = 'levels/'
file='testing_levels.txt'
title = 'RMSE - Varying levels'

levels = [2, 4, 6, 8, 10, 12]
shape = (len(levels),)

final_matrices={'Full Non Deformed': {'rmse' : np.zeros(shape)}, 
                'Full Deformed': {'rmse' : np.zeros(shape)}, 
                'Partial Deformed': {'rmse' : np.zeros(shape)},  
                'Partial Non Deformed': {'rmse' : np.zeros(shape)}}

file_txt = open(base + folder + file, 'r')
Lines = file_txt.readlines()
levels_val = -1
current_data_type = ''
for line in Lines:
    if line[:-1] in data_types:
        current_data_type = line[:-1]
    if 'Test - levels' in line:
        levels_val = int(re.findall('-?\d+', line)[0])
    if 'RMSE' in line:
        rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = levels.index(levels_val)
        final_matrices[current_data_type]['rmse'][i] = rmse
        
print('final_matrices : ', final_matrices)

for data_type in data_types:
    plt.clf()
    levels_po = range(len(levels))
    plt.clf()
    plt.title(title)
    plt.plot(levels_po, final_matrices[data_type]['rmse'], color='r')
    plt.xticks(levels_po, levels, rotation=90)
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_graph.png', bbox_inches='tight')