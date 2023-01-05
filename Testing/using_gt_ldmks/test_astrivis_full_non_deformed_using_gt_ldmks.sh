type=kpfcn
# type=fcgf

filename=Testing/using_gt_ldmks/test_astrivis_full_non_deformed_gt_ldmks.txt
folder_name=output_full_non_deformed_gt_ldmks
rm ${filename}
touch ${filename}

training_data=full_deformed
# training_data=partial_deformed
# training_data=pretrained

if [ "$type" == "kpfcn" ] ; then
	config=LNDP.yaml
else
	config=LNDP_fcgf.yaml
fi

if [ "$training_data" == "full_deformed" ] ; then
	epoch=10
elif [ "$training_data" == "partial_deformed" ] ; then
	epoch=5
elif [ "$training_data" == "pretrained" ] ; then
	epoch=null
fi

base='/home/aiday.kyzy/dataset/Synthetic/FullNonDeformedData/TestingData/'
# model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
model_numbers=('002' '042' '085' '126' '167' '207')

for k in ${model_numbers[@]}
do
    echo "model ${k}" >> ${filename}
    mkdir $base/model$k/${folder_name}
    touch ${base}/model${k}/${folder_name}/0_1_se4.h5
    
    python3 eval_supervised_astrivis.py \
    --config=config/${config} \
    --s="model${k}/mesh_transformed_0.ply" \
    --t="model${k}/mesh_transformed_1.ply" \
    --source_trans="model${k}/mesh_transformed_0_se4.h5" \
    --target_trans="model${k}/mesh_transformed_1_se4.h5" \
    --matches="model${k}/0_1.npz" \
    --output="model${k}/${folder_name}/0_1.ply" \
    --output_trans="model${k}/${folder_name}/0_1_se4.h5" \
    --intermediate_output_folder="model${k}/${folder_name}/" \
    --base=${base} \
    --print_keypoints \
    --use_gt_ldmks \
    >> ${filename}
    
    if [ "$?" != "1" ]; then
    python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
    --part1="${base}/model${k}/mesh_transformed_0_se4.h5" \
    --part2="${base}/model${k}/mesh_transformed_1_se4.h5" \
    --pred="${base}/model${k}/${folder_name}/0_1_se4.h5" >> ${filename}
    
    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
    --input1="${base}/model${k}/${folder_name}/0_1.ply" \
    --input2="${base}/model${k}/mesh_transformed_1.ply" \
    --matches="${base}/model${k}/0_1.npz" >> ${filename}
    fi
done