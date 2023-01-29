import matplotlib.pyplot as plt
import numpy as np

full_deformed_rmse = [0.01307, 0.06942, 0.05478, 0.01970, 0.01976, 0.04744]
partial_deformed_rmse = [0.11546, 0.11725, 0.11768, 0.13169, 0.14505, 0.10239]
full_non_deformed_rmse = [0.02725, 0.004741, 0.01228, 0.01251, 0.00683, 0.13166]
partial_non_deformed_rmse = [0.15101, 0.03149, 0.09218, 0.07869, 0.142086, 0.071286]

models = ['002', '042', '085', '126', '167', '207']
bar = np.array([0, 1, 2, 3, 4, 5])

plt.clf()
plt.bar(bar, full_deformed_rmse)
plt.xlabel('Model number')
plt.ylabel('RMSE')
plt.xticks(bar, models)
plt.title('Full Deformed')
plt.savefig('Testing/all/full_deformed_initial_testing.png')

plt.clf()
plt.bar(bar, partial_deformed_rmse)
plt.xlabel('Model number')
plt.ylabel('RMSE')
plt.xticks(bar, models)
plt.title('Partial Deformed')
plt.savefig('Testing/all/partial_deformed_initial_testing.png')

plt.clf()
plt.bar(bar, full_non_deformed_rmse)
plt.xlabel('Model number')
plt.ylabel('RMSE')
plt.xticks(bar, models)
plt.title('Full Non Deformed')
plt.savefig('Testing/all/full_non_deformed_initial_testing.png')

plt.clf()
plt.bar(bar, partial_non_deformed_rmse)
plt.xlabel('Model number')
plt.ylabel('RMSE')
plt.xticks(bar, models)
plt.title('Partial Non Deformed')
plt.savefig('Testing/all/partial_non_deformed_initial_testing.png')