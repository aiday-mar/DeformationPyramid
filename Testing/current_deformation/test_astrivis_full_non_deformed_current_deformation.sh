type=fcgf
# type=kpfcn

preprocessing=none
# preprocessing=mutual

training_data=full_deformed
# training_data=partial_deformed
# training_data=pretrained

# knn_matching=True
knn_matching=False

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
	# confidence_threshold=0.1
    # confidence_threshold_name=0.1
    confidence_threshold=0.01
	confidence_threshold_name=0.01
else
    confidence_threshold=0.01
	confidence_threshold_name=1e-02
	# confidence_threshold=0.000001
    # confidence_threshold_name=1e-06
fi

model_numbers=('002' '042' '085' '126' '167' '207')
# model_numbers=('085')

# one_model=True
one_model=False

if [ "$one_model" == "False" ] ; then
	filename=Testing/current_deformation/test_astrivis_full_non_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_knn_${knn_matching}_conf_${confidence_threshold_name}.txt
	folder_name=output_full_non_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_knn_${knn_matching}_conf_${confidence_threshold_name}
fi

if [ "$one_model" == "True" ] ; then
	filename=Testing/current_deformation/test_astrivis_full_non_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_knn_${knn_matching}_conf_${confidence_threshold_name}_one_model.txt
	folder_name=output_full_non_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_knn_${knn_matching}_conf_${confidence_threshold_name}
fi

rm ${filename}
touch ${filename}

base='/home/aiday.kyzy/dataset/Synthetic/FullNonDeformedData/TestingData/'

if [ $knn_matching == "False" ]; then
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
            --base=$base \
            --confidence_threshold=${confidence_threshold} \
            --only_inference \
            --preprocessing=${preprocessing} \
            --print_keypoints >> ${filename}
            
            if [ "$?" != "1" ]; then
            rm "${base}/model${k}/${folder_name}/current_deformation.ply"

            python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
            --file1="${base}/model${k}/mesh_transformed_0.ply" \
            --file2="${base}/model${k}/mesh_transformed_1.ply" \
            --landmarks1="${base}/model${k}/${folder_name}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
            --landmarks2="${base}/model${k}/${folder_name}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
            --save_path="${base}/model${k}/${folder_name}/current_deformation.ply" >> ${filename}
            
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --input1="${base}/model${k}/${folder_name}/current_deformation.ply" \
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
            --base=$base \
            --confidence_threshold=${confidence_threshold} \
            --only_inference \
            --preprocessing=${preprocessing} \
            --print_keypoints >> ${filename}
            
            if [ "$?" != "1" ]; then
            rm "${base}/model${k}/${folder_name}/current_deformation.ply" >> ${filename}
            
            python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
            --file1="${base}/model${k}/mesh_transformed_0.ply" \
            --file2="${base}/model${k}/mesh_transformed_1.ply" \
            --landmarks1="${base}/model${k}/${folder_name}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
            --landmarks2="${base}/model${k}/${folder_name}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
            --save_path="${base}/model${k}/${folder_name}/current_deformation.ply" >> ${filename}
            
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --input1="${base}/model${k}/${folder_name}/current_deformation.ply" \
            --input2="${base}/model${k}/mesh_transformed_1.ply" \
            --matches="${base}/model${k}/0_1.npz" >> ${filename}
            fi
        done
    fi
fi

if [ $knn_matching == "True" ]; then
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
            --base=$base \
            --confidence_threshold=${confidence_threshold} \
            --only_inference \
            --preprocessing=${preprocessing} \
            --knn_matching \
            --print_keypoints >> ${filename}
            
            if [ "$?" != "1" ]; then
            rm "${base}/model${k}/${folder_name}/current_deformation.ply"

            python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
            --file1="${base}/model${k}/mesh_transformed_0.ply" \
            --file2="${base}/model${k}/mesh_transformed_1.ply" \
            --landmarks1="${base}/model${k}/${folder_name}/${type}_ldmk/s_knn_matching_pcd.ply" \
            --landmarks2="${base}/model${k}/${folder_name}/${type}_ldmk/t_knn_matching_pcd.ply" \
            --save_path="${base}/model${k}/${folder_name}/current_deformation.ply" >> ${filename}
            
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --input1="${base}/model${k}/${folder_name}/current_deformation.ply" \
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
            --base=$base \
            --confidence_threshold=${confidence_threshold} \
            --only_inference \
            --preprocessing=${preprocessing} \
            --knn_matching \
            --print_keypoints >> ${filename}
            
            if [ "$?" != "1" ]; then
            rm "${base}/model${k}/${folder_name}/current_deformation.ply" >> ${filename}
            
            python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
            --file1="${base}/model${k}/mesh_transformed_0.ply" \
            --file2="${base}/model${k}/mesh_transformed_1.ply" \
            --landmarks1="${base}/model${k}/${folder_name}/${type}_ldmk/s_knn_matching_pcd.ply" \
            --landmarks2="${base}/model${k}/${folder_name}/${type}_ldmk/t_knn_matching_pcd.ply" \
            --save_path="${base}/model${k}/${folder_name}/current_deformation.ply" >> ${filename}
            
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --input1="${base}/model${k}/${folder_name}/current_deformation.ply" \
            --input2="${base}/model${k}/mesh_transformed_1.ply" \
            --matches="${base}/model${k}/0_1.npz" >> ${filename}
            fi
        done
    fi
fi