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
folder = 'posenc_function/'
file='testing_posenc_functions.txt'
title = 'RMSE - Varying posenc functions'

posenc_functions = ['none', 'power4', 'log', 'linear', 'square']
shape = (len(posenc_functions),)

final_matrices={'Full Non Deformed': {'rmse' : np.zeros(shape)}, 
                'Full Deformed': {'rmse' : np.zeros(shape)}, 
                'Partial Deformed': {'rmse' : np.zeros(shape)},  
                'Partial Non Deformed': {'rmse' : np.zeros(shape)}}

file_txt = open(base + folder + file, 'r')
Lines = file_txt.readlines()
posenc_function_val = ''
current_data_type = ''
for line in Lines:
    if line[:-1] in data_types:
        current_data_type = line[:-1]
    if 'Test - levels' in line:
        interm_list = line.split()
        print(interm_list)
        posenc_function_val = interm_list[-1][:-1]
    if 'RMSE' in line:
        rmse = list(map(float, re.findall("\d+\.\d+", line)))[0]
        i = posenc_functions.index(posenc_function_val)
        final_matrices[current_data_type]['rmse'][i] = rmse
        
print('final_matrices : ', final_matrices)

for data_type in data_types:
    plt.clf()
    posenc_functions_pos = range(len(posenc_functions))
    plt.clf()
    plt.title(title)
    plt.plot(posenc_functions_pos, final_matrices[data_type]['rmse'], color='r')
    plt.xticks(posenc_functions_pos, posenc_functions, rotation=90)
    plt.savefig(base + folder + data_type.replace(' ', '_') + '_graph.png', bbox_inches='tight')