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

confidence_thresholds = [0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]
shape = (len(confidence_thresholds),)

final_matrices={'Full Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}, 
                'Full Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}, 
                'Partial Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}},  
                'Partial Non Deformed': {'lepard' : {'total' : np.zeros(shape), 'true' : np.zeros(shape)}}}

file='confidence_threshold/testing_confidence_thresholds.txt'

for i in range(len(confidence_thresholds)):
    confidence_threshold = confidence_thresholds[i]
    files.append(file)
    file_txt = open(base + file, 'r')
    Lines = file_txt.readlines()
    current_data_type = ''
    for line in Lines:
        if line[:-1] in data_types:
            current_data_type = line[:-1]

        if 'number of true landmarks correspondences returned from Lepard' in line:
            search = list(map(int, re.findall(r'\d+', line)))
            true = int(search[0])
            total = int(search[1])
            final_matrices[current_data_type]['lepard']['true'][i] = true
            final_matrices[current_data_type]['lepard']['total'][i] = total - true

for data_type in data_types:
    plt.clf()
    true_data = []
    total_data = []
    fraction = []
    
    for i in range(len(confidence_thresholds)):
                
        true_data.append(final_matrices[data_type]['lepard']['true'][i])
        total_data.append(final_matrices[data_type]['lepard']['total'][i])
          
        if final_matrices[data_type]['lepard']['total'][i] != 0:
            fraction.append(final_matrices[data_type]['lepard']['true'][i]/(final_matrices[data_type]['lepard']['total'][i]+final_matrices[data_type]['lepard']['true'][i]))
        else:
            fraction.append(0)

    plt.title('Varying the confidence threshold')
    confidence_thresholds_pos = range(0, len(confidence_thresholds))
    plt.bar(confidence_thresholds_pos, true_data, color='r')
    plt.bar(confidence_thresholds_pos, total_data, bottom=true_data, color='b')
    plt.xticks(confidence_thresholds_pos, confidence_thresholds, rotation=90)
    plt.savefig('Testing/confidence_threshold/' + data_type.replace(' ', '_'), bbox_inches='tight')
    
    plt.clf()
    plt.title('Varying the confidence threshold')
    plt.plot(confidence_thresholds_pos, fraction, color='r')
    plt.xticks(confidence_thresholds_pos, confidence_thresholds, rotation=90)
    plt.ylim(0, 1)
    plt.savefig('Testing/confidence_threshold/' + data_type.replace(' ', '_'), bbox_inches='tight')