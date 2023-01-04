# config=LNDP_fcgf.yaml
config=LNDP.yaml

# type=fcgf
type=kpfcn

# preprocessing=none
preprocessing=mutual

# training_data=full_deformed
# training_data=partial_deformed
training_data=pretrained

# epoch=2
# epoch=1
epoch=null
# epoch=5

if [ "$training_data" == "pretrained" ] ; then
	confidence_threshold=0.1
else
	confidence_threshold=0.000001
fi

filename=Testing/current_deformation/test_astrivis_partial_non_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}.txt
folder_name=output_partial_non_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}
rm ${filename}
touch ${filename}

base='/home/aiday.kyzy/dataset/Synthetic/PartialNonDeformedData/TestingData/'
model_numbers=('002' '042' '085' '126' '167' '207')

if [ $type == "kpfcn" ]; then
    for k in ${model_numbers[@]}
    do
        mkdir $base/model$k/${folder_name}
        echo "model ${k}" >> ${filename}

        python3 eval_supervised_astrivis.py \
        --config=config/${config} \
        --s="model${k}/transformed/mesh_transformed_0.ply" \
        --t="model${k}/transformed/mesh_transformed_1.ply" \
        --source_trans="model${k}/transformed/mesh_transformed_0_se4.h5" \
        --target_trans="model${k}/transformed/mesh_transformed_1_se4.h5" \
        --matches="model${k}/matches/0_1.npz" \
        --output="model${k}/${folder_name}/0_1.ply" \
        --output_trans="model${k}/${folder_name}/0_1_se4.h5" \
        --intermediate_output_folder="model${k}/${folder_name}/" \
        --base=$base \
        --confidence_threshold=${confidence_threshold} \
        --only_inference \
        --preprocessing=${preprocessing} \
        --print_keypoints >> ${filename}
        
        if [ "$?" != "1" ]; then
        python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
        --file1="${base}/model${k}/transformed/mesh_transformed_0.ply" \
        --file2="${base}/model${k}/transformed/mesh_transformed_1.ply" \
        --landmarks1="${base}/model${k}/${folder_name}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
        --landmarks2="${base}/model${k}/${folder_name}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
        --save_path="${base}/model${k}/${folder_name}/current_deformation.ply" >> ${filename}

        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
        --final="${base}/model${k}/${folder_name}/current_deformation.ply" \
        --initial="${base}model${k}/transformed/mesh_transformed_0.ply" \
        --part1="${base}model${k}/transformed/mesh_transformed_0_se4.h5" \
        --part2="${base}model${k}/transformed/mesh_transformed_1_se4.h5" >> ${filename}
        fi
    done
fi

if [ $type == "fcgf" ]; then
    for k in ${model_numbers[@]}
    do
        mkdir $base/model$k/${folder_name}
        echo "model ${k}" >> ${filename}

        python3 eval_supervised_astrivis.py \
        --config=config/${config} \
        --s="model${k}/transformed/mesh_transformed_0.ply" \
        --t="model${k}/transformed/mesh_transformed_1.ply" \
        --s_feats="model${k}/transformed/mesh_transformed_0_fcgf.npz" \
        --t_feats="model${k}/transformed/mesh_transformed_1_fcgf.npz" \
        --source_trans="model${k}/transformed/mesh_transformed_0_se4.h5" \
        --target_trans="model${k}/transformed/mesh_transformed_1_se4.h5" \
        --matches="model${k}/matches/0_1.npz" \
        --output="model${k}/${folder_name}/0_1.ply" \
        --output_trans="model${k}/${folder_name}/0_1_se4.h5" \
        --intermediate_output_folder="model${k}/${folder_name}/" \
        --base=$base \
        --confidence_threshold=${confidence_threshold} \
        --only_inference \
        --preprocessing=${preprocessing} \
        --print_keypoints >> ${filename}
        
        if [ "$?" != "1" ]; then
        python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
        --file1="${base}/model${k}/transformed/mesh_transformed_0.ply" \
        --file2="${base}/model${k}/transformed/mesh_transformed_1.ply" \
        --landmarks1="${base}/model${k}/${folder_name}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
        --landmarks2="${base}/model${k}/${folder_name}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
        --save_path="${base}/model${k}/${folder_name}/current_deformation.ply" >> ${filename}

        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
        --final="${base}/model${k}/${folder_name}/current_deformation.ply" \
        --initial="${base}model${k}/transformed/mesh_transformed_0.ply" \
        --part1="${base}model${k}/transformed/mesh_transformed_0_se4.h5" \
        --part2="${base}model${k}/transformed/mesh_transformed_1_se4.h5" >> ${filename}
        fi
    done
fi