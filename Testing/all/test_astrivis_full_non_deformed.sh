config=LNDP_fcgf.yaml
# config=LNDP.yaml

type=fcgf
# type=kpfcn

filename=Testing/all/test_astrivis_full_non_deformed_${type}.txt
rm ${filename}
touch ${filename}

base='/home/aiday.kyzy/dataset/Synthetic/FullNonDeformedData/TestingData/'
# model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
model_numbers=('002' '042' '085' '126' '167' '207')
folder_name=output_full_non_deformed_${type}

for k in ${model_numbers[@]}
do
    echo "model ${k}" >> ${filname}
    mkdir $base/model$k/${folder_name}
    touch ${base}/model${k}/${folder_name}/0_1_se4.h5
    
    python3 eval_supervised_astrivis.py \
    --config=config/${config} \
    --s="model${k}/mesh_transformed_0.ply" \
    --t="model${k}/mesh_sampled.ply" \
    --source_trans="model${k}/mesh_transformed_0_se4.h5" \
    --target_trans="identity.h5" \
    --matches="model${k}/0_1.npz" \
    --output="model${k}/${folder_name}/0_1.ply" \
    --output_trans="model${k}/${folder_name}/0_1_se4.h5" \
    --intermediate_output_folder="model${k}/${folder_name}/" \
    --base=${base} \
    --print_keypoints >> ${filname}
    
    if [ "$?" != "1" ]; then
    python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
    --part1="${base}/model${k}/mesh_transformed_0_se4.h5" \
    --part2="identity.h5" 
    --pred="${base}/model${k}/${folder_name}/0_1_se4.h5" >> ${filname}
    
    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
    --input1="${base}/model${k}/${folder_name}/0_1.ply" \
    --input2="${base}/model${k}/mesh_sampled.ply" \
    --matches="${base}/model${k}/0_1.npz" >> ${filname}
    fi
done