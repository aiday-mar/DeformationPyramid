src_pcd_points = data['src_pcd_list'][0]
n_src_points = src_pcd_points.shape[0]
dists = np.zeros((n_src_points, n_src_points))
for i in range(n_src_points):
    src_point = src_pcd_points[i]
    difference = src_pcd_points - src_point
    square_difference = difference ** 2
    square_difference = np.array(square_difference.cpu())
    sum_by_row = np.sum(square_difference, axis=1)
    dists[i, :] = np.sqrt(sum_by_row)

for i in range(n_src_points):
    dists[i][i] = float('inf')
min_distances = dists.min(axis = 1)
average_distance = np.average(min_distances)
neighbors = dists < 1.5 * average_distance
n_neighbors = np.sum(neighbors, axis=1)
initial_edge_point_indices = np.argwhere(n_neighbors < 3)