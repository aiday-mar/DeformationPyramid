import re
import matplotlib.pyplot as plt
import numpy as np
import copy

w_reg_list = [0, 0.3, 0.5, 0.7, 1]
w_cd_list = [0, 0.3, 0.5, 0.7, 1]
types=['Full Non Deformed', 'Full Deformed', 'Partial Deformed', 'Partial Non Deformed']
matrix = np.zeros([len(w_reg_list), len(w_cd_list)])
final_matrices = {type : copy.deepcopy(matrix) for type in types}
w_reg_idx = 0
w_cd_idx = 0
# model='042'
model='002'

file='Testing/w_cd_w_reg/testing_w_pre_mutual_kpfcn_td_pretrained_' + model + '.txt'
f = open(file, "r")
idx = 0
for line in f:
    if 'w_cd' in line:
        list_res = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
        res = float(list_res[0])
        w_cd_idx = w_cd_list.index(res)
    if 'w_reg' in line:
        idx = 0
        list_res = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
        res = float(list_res[0])
        w_reg_idx = w_reg_list.index(res)
    if 'RMSE' in line and w_reg_idx is not None and w_cd_idx is not None:
        list_res = re.findall("\d+\.\d+", line)
        res = float(list_res[0])
        final_matrices[types[idx]][w_reg_idx][w_cd_idx] = res
        idx += 1

for idx in range(len(types)):
                    
    x = ["0", "0.3", "0.5", "0.7", "1"]
    y = ["0", "0.3", "0.5", "0.7", "1"]
    
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(final_matrices[types[idx]])

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
            text = ax.text(j, i, format(final_matrices[types[idx]][i, j], '.5f'),
                        ha="center", va="center", color="w")
    type = types[idx]
    ax.set_title(type)
    fig.tight_layout()
    fig.savefig('Testing/w_cd_w_reg/' + types[idx].replace(' ', '_') + '_rmse_' + model + '.png')
                   
    