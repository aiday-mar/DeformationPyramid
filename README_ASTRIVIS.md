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
