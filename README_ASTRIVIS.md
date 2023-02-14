This branch is used for testing NDP. To test NDP you may go into the Testing folder which contains several sub-folders for the different tests performed. The tests are described below per folder:

```
all: General testing of NDP. It contains 4 shell files for testing NDP on the 4 categories of data and 4 more shell files for testing NDP with the custom filtering technique

confidence_threshold: testing the effect of the confidence threshold on the RMSE

current_deformation: testing the pipeline which uses the astrivis deformation algorithm instead of the deformation algorithm from NDP

custom_filtering: optimizing the parameters of the custom filtering

exterior_boundary_detection: testing the RMSE when using the exterior boundary detection filtering for partial scans

exterior_boundary_detection_result: visualizing the detected boundary edges for partial scans

k0: testing the effect of k0 on the RMSE

levels: testing the effect of the number of NDP levels on the RMSE

posenc_function: testing the effect of different positional encoding function on the RMSE

samples: testing the effect of different numbers of samples on the RMSE

using_gt_ldmks: testing the case when the ground-truth landmarks are used in NDP

w_cd_w_reg: testing different regularization and chamfer distance weights
```

The tests are all run from the shell files. These call the eval_supervised_astrivis.py file with different flags as follows:

```
s
t
s_feats
t_feats
output
output_trans
matches
source_trans
target_trans
config
base
w_cd
k0
w_reg
confidence_threshold
samples
levels
posenc_function
reject_outliers
intermediate_output_folder
preprocessing
coarse_level
index_coarse_feats
number_centers
average_distance_multiplier
number_iterations_custom_filtering
inlier_outlier_thr
mesh_path
sampling
max_ldmks
indent
gt_thr
edge_filtering_simple
edge_filtering_angle
edge_filtering_shape
edge_filtering_disc
edge_filtering_mesh
min_dist_thr
n_points_edge_filtering
visualize
custom_filtering
print_keypoints
use_gt_ldmks
only_inference
knn_matching
```
