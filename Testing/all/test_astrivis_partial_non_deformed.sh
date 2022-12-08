config=LNDP_fcgf.yaml
# config=LNDP.yaml

type=fcgf
# type=kpfcn

filename=Testing/all/test_astrivis_partial_non_deformed_${type}.txt
rm ${filename}
touch ${filename}

base='/home/aiday.kyzy/dataset/Synthetic/PartialNonDeformedData/TestingData'

# model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
model_numbers=('002' '042' '085' '126' '167' '207')
folder_name=output_partial_non_deformed_${type}

for k in ${model_numbers[@]}
do

    mkdir $base/model$k/${folder_name}
    echo "model ${k}" >> ${filename}

    # 0 -> 1
    touch ${base}/model${k}/${folder_name}/0_1_se4.h5
    python3 eval_supervised_astrivis.py \
    --config=config/${config} \
    --s="PartialNonDeformedData/TestingData/model${k}/transformed/mesh_transformed_0.ply" \
    --t="PartialNonDeformedData/TestingData/model${k}/temp/model_1/cloud/dense.ply" \
    --source_trans="PartialNonDeformedData/TestingData/model${k}/transformed/mesh_transformed_0_se4.h5" \
    --target_trans="identity.h5" \
    --matches="PartialNonDeformedData/TestingData/model${k}/matches/0_1.npz" \
    --output="PartialNonDeformedData/TestingData/model${k}/${folder_name}/0_1.ply" \
    --output_trans="PartialNonDeformedData/TestingData/model${k}/${folder_name}/0_1_se4.h5" \
    --intermediate_output_folder="PartialNonDeformedData/TestingData/model${k}/${folder_name}/" \
    --base=${base} \
    --print_keypoints >> ${filename}
    
    if [ "$?" != "1" ]; then
    python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
    --part1="${base}/model${k}/transformed/mesh_transformed_0_se4.h5" \
    --part2="identity.h5" \
    --pred="${base}/model${k}/${folder_name}/0_1_se4.h5" >> ${filename}

    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
    --input1="${base}/model${k}/${folder_name}/0_1.ply" \
    --input2="${base}/model${k}/temp/model_0/cloud/dense.ply" >> ${filename}
    fi
done