# type=fcgf
type=kpfcn

# preprocessing=none
preprocessing=mutual

# training_data=full_deformed
# training_data=partial_deformed
training_data=pretrained

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

if [ "$training_data" == "pretrained" ] ; then
	confidence_threshold=0.1
else
	confidence_threshold=0.000001
fi

filename=Testing/all/test_astrivis_full_non_deformed_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}.txt
folder_name=output_full_non_deformed_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}
rm ${filename}
touch ${filename}

base='/home/aiday.kyzy/dataset/Synthetic/FullNonDeformedData/TestingData/'
model_numbers=('002' '042' '085' '126' '167' '207')
n_deformed_levels=4
n_non_deformed_levels=1
w_cd=0
w_reg=0

if [ $type == "kpfcn" ]; then
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
        --confidence_threshold=${confidence_threshold} \
        --preprocessing=${preprocessing} \
        --level=${n_non_deformed_levels} \
        --w_cd=${w_cd} \
		--w_reg=${w_reg} \
        --print_keypoints >> ${filename}
        
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
fi

if [ $type == "fcgf" ]; then
    for k in ${model_numbers[@]}
    do
        echo "model ${k}" >> ${filename}
        mkdir $base/model$k/${folder_name}
        touch ${base}/model${k}/${folder_name}/0_1_se4.h5
        
        python3 eval_supervised_astrivis.py \
        --config=config/${config} \
        --s="model${k}/mesh_transformed_0.ply" \
        --t="model${k}/mesh_transformed_1.ply" \
        --s_feats="model${k}/mesh_transformed_0_fcgf.npz" \
        --t_feats="model${k}/mesh_transformed_1_fcgf.npz" \
        --source_trans="model${k}/mesh_transformed_0_se4.h5" \
        --target_trans="model${k}/mesh_transformed_1_se4.h5" \
        --matches="model${k}/0_1.npz" \
        --output="model${k}/${folder_name}/0_1.ply" \
        --output_trans="model${k}/${folder_name}/0_1_se4.h5" \
        --intermediate_output_folder="model${k}/${folder_name}/" \
        --base=${base} \
        --confidence_threshold=${confidence_threshold} \
        --preprocessing=${preprocessing} \
        --level=${n_non_deformed_levels} \
        --w_cd=${w_cd} \
		--w_reg=${w_reg} \
        --print_keypoints >> ${filename}
        
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
fi