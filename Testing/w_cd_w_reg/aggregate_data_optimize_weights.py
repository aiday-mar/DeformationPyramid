import re
import matplotlib.pyplot as plt
import numpy as np

w_reg_list = [0, 0.3, 0.5, 0.7, 1]
w_cd_list = [0, 0.3, 0.5, 0.7, 1]
types=['Full Deformed', 'Full Non Deformed', 'Partial Deformed', 'Partial Non Deformed']
matrix = np.empty([len(w_reg_list), len(w_cd_list)])

w_reg_idx = 0
w_cd_idx = 0

file='Testing/w_cd_w_reg/testing_w_pre_mutual_kpfcn_td_pretrained.txt'
f = open(file, "r")
for line in f:
    if 'w_cd' in line:
        list_res = re.findall("\d+\.\d+", line)
        res = float(list_res[0])
        w_cd_idx = w_cd_list.index(res)
    if 'w_reg' in line:
        list_res = re.findall("\d+\.\d+", line)
        res = float(list_res[0])
        w_reg_idx = w_reg_list.index(res)
    if 'RMSE' in line and w_reg_idx is not None and w_cd_idx is not None:
        list_res = re.findall("\d+\.\d+", line)
        res = float(list_res[0])
        matrix[w_reg_idx][w_cd_idx] = res
        break
    
for idx in range(len(types)):
                    
    x = ["0", "0.3", "0.5", "0.7", "1"]
    y = ["0", "0.3", "0.5", "0.7", "1"]
    
    plt.clf()
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
    type = types[idx]
    ax.set_title(type)
    fig.tight_layout()
    fig.savefig('Testing/w_cd_w_reg/' + types[idx].replace(' ', '_') + '_rmse.png')
                   
    