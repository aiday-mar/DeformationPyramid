base='/home/aiday.kyzy/dataset/Synthetic/PartialNonDeformedData/TestingData'

model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
for k in ${model_numbers[@]}
do

    mkdir $base/model$k/output
    touch ${base}/model${k}/output/0_1_se4.h5
    python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s="PartialNonDeformedData/TestingData/model${k}/transformed/mesh_transformed_0.ply" --t="PartialNonDeformedData/TestingData/model${k}/transformed/mesh_transformed_1.ply" --source_trans="PartialNonDeformedData/TestingData/model${k}/transformed/mesh_transformed_0_se4.h5" --target_trans="PartialNonDeformedData/TestingData/model${k}/transformed/mesh_transformed_1_se4.h5" --matches="PartialNonDeformedData/TestingData/model${k}/matches/0_1.npz" --output="PartialNonDeformedData/TestingData/model${k}/output/0_1.ply" --output_trans="PartialNonDeformedData/TestingData/model${k}/output/0_1_se4.h5"
    python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="${base}/model${k}/transformed/mesh_transformed_0_se4.h5" --part2="${base}/model${k}/transformed/mesh_transformed_1_se4.h5" --pred="${base}/model${k}/output/0_1_se4.h5"
    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="${base}/model${k}/output/0_1.ply" --initial="${base}/model${k}/transformed/mesh_transformed_0.ply" --part1="${base}/model${k}/transformed/mesh_transformed_0_se4.h5" --part2="${base}/model${k}/transformed/mesh_transformed_1_se4.h5"

done