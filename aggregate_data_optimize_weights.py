import re
import matplotlib.pyplot as plt
import numpy as np

w_reg_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
w_cd_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
base = 'TestData/'
types=['FullDeformed/', 'FullNonDeformed/', 'PartialDeformed/', 'PartialNonDeformed/']
metric_list=['RMSE', 'IR', 'full-epe', 'full-AccR', 'full-AccS', 'full-outlier', 'vis-epe', 'vis-AccR', 'vis-AccS', 'vis-outlier']
metric = 'RMSE'
matrix = np.empty([len(w_reg_list), len(w_cd_list)])

for type in types:
    path = base + type
    for w_reg_idx in range(0, len(w_reg_list)):
        for w_cd_idx in range(0, len(w_cd_list)):
            w_reg = w_reg_list[w_reg_idx]
            w_cd = w_cd_list[w_cd_idx]
            result_path = path + 'output_' + str(w_reg) + '_' + str(w_cd) + '/result.txt'
            f = open(result_path, "r")
            for line in f:
                if metric in line:
                    list_res = re.findall("\d+\.\d+", line)
                    res = float(list_res[0])
                    matrix[w_reg_idx][w_cd_idx] = res
                    break

plt.imshow(matrix, cmap='hot', interpolation='nearest')
plt.show()
                   
    