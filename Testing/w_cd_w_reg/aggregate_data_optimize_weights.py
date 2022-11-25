import re
import matplotlib.pyplot as plt
import numpy as np

w_reg_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
w_cd_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
base = '../../TestData/'
types=['FullDeformed/', 'FullNonDeformed/', 'PartialDeformed/', 'PartialNonDeformed/']
names=['Full Deformed', 'Full Non Deformed', 'Partial Deformed', 'Partial Non Deformed']
metric_list=['RMSE', 'IR', 'full-epe', 'full-AccR', 'full-AccS', 'full-outlier', 'vis-epe', 'vis-AccR', 'vis-AccS', 'vis-outlier']

for metric in ['RMSE', 'IR']:
    for idx in range(len(types)):
        type = types[idx]
        matrix = np.empty([len(w_reg_list), len(w_cd_list)])
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
        
        x = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
        y = ["0", "0.2", "0.4", "0.6", "0.8", "1"]

        fig, ax = plt.subplots()
        im = ax.imshow(matrix)

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('', rotation=-90, va="bottom")

        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        plt.xlabel('w_reg')
        ax.set_yticks(np.arange(len(y)))
        ax.set_yticklabels(y)
        plt.ylabel('w_cd')

        for i in range(len(x)):
            for j in range(len(y)):
                text = ax.text(j, i, format(matrix[i, j], '.5f'),
                            ha="center", va="center", color="w")

        ax.set_title(metric + ' - ' + names[idx])
        fig.tight_layout()
        fig.savefig(metric + '_' + names[idx].replace(' ', '_'))
                   
    