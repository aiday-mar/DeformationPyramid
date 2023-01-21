
# criterion=none
# criterion=simple
# criterion=angle
# criterion=shape
# criterion=disc
criterion=mesh

type=fcgf
# type=kpfcn

preprocessing=none
# preprocessing=mutual

training_data=partial_deformed
# training_data=pretrained

current_deformation=True
# current_deformation=False

n_deformed_levels=8
n_non_deformed_levels=1

if [ "$type" == "kpfcn" ] ; then
	config=LNDP.yaml
    min_dist_thr_kpfcn=0.01
else
	config=LNDP_fcgf.yaml
    min_dist_thr_fcgf=0.01
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

if [ $criterion == "simple" ]; then
    edge_filtering_type=edge_filtering_simple
fi

if [ $criterion == "angle" ]; then
    edge_filtering_type=edge_filtering_angle
fi

if [ $criterion == "shape" ]; then
    edge_filtering_type=edge_filtering_shape
fi

if [ $criterion == "disc" ]; then
    edge_filtering_type=edge_filtering_disc
fi

if [ $criterion == "mesh" ]; then
    edge_filtering_type=edge_filtering_mesh
fi

model_numbers=('002' '042' '085' '126' '167' '207')
base='/home/aiday.kyzy/code/DeformationPyramid/TestData/'

if [ $current_deformation == "False" ]; then

    file="Testing/exterior_boundary_detection/testing_${criterion}_edge_filtering_pre_${preprocessing}_${type}_td_${training_data}_epoch_${epoch}.txt"
    rm ${file}
    touch ${file}

    if [ $type == "kpfcn" ]; then
        for k in ${model_numbers[@]}
        do
            echo "model ${k}" >> ${file}
            folder="${criterion}_edge_filtering_pre_${preprocessing}_${type}_td_${training_data}_epoch_${epoch}"
            rm -rf TestData/PartialNonDeformed/model${k}/${folder}
            mkdir TestData/PartialNonDeformed/model${k}/${folder}
            rm -rf TestData/PartialDeformed/model${k}/${folder}
            mkdir TestData/PartialDeformed/model${k}/${folder}

            if [ $criterion == "none" ]; then

                echo 'Partial Deformed' >> ${file}
                echo 'Edge filtering not used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialDeformed/model${k}/020_0.ply" \
                --t="PartialDeformed/model${k}/104_1.ply" \
                --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
                --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
                --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
                --output="PartialDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --preprocessing=${preprocessing} \
                --level=${n_deformed_levels} \
                --print_keypoints >> ${file}
                
                if [ "$?" != "1" ]; then

                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialDeformed/model${k}/${folder}/result.ply" \
                    --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
                    --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
                    --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
                    --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                    --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                    --save_final_path="TestData/PartialDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi

                echo 'Partial Non Deformed' >> ${file}
                echo 'Edge filtering not used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                --t="PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                --source_trans="PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --target_trans="PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --matches="PartialNonDeformed/model${k}/0_1.npz" \
                --output="PartialNonDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialNonDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialNonDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --preprocessing=${preprocessing} \
                --level=${n_non_deformed_levels} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                
                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialNonDeformed/model${k}/${folder}/result.ply" \
                    --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                    --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                    --save_final_path="TestData/PartialNonDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi
                
            else

                echo 'Partial Deformed' >> ${file}
                echo 'Edge filtering used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialDeformed/model${k}/020_0.ply" \
                --t="PartialDeformed/model${k}/104_1.ply" \
                --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
                --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
                --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
                --output="PartialDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --${edge_filtering_type} \
                --reject_outliers=false \
                --min_dist_thr=${min_dist_thr_kpfcn} \
                --preprocessing=${preprocessing} \
                --level=${n_deformed_levels} \
                --print_keypoints >> ${file}
                
                if [ "$?" != "1" ]; then
                    
                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialDeformed/model${k}/${folder}/result.ply" \
                    --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
                    --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
                    --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
                    --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                    --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                    --save_final_path="TestData/PartialDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi

                echo 'Partial Non Deformed' >> ${file}
                echo 'Edge filtering used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                --t="PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                --source_trans="PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --target_trans="PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --matches="PartialNonDeformed/model${k}/0_1.npz" \
                --output="PartialNonDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialNonDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialNonDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --${edge_filtering_type} \
                --reject_outliers=false \
                --min_dist_thr=${min_dist_thr_kpfcn} \
                --preprocessing=${preprocessing} \
                --level=${n_non_deformed_levels} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                    
                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialNonDeformed/model${k}/${folder}/result.ply" \
                    --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                    --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                    --save_final_path="TestData/PartialNonDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi
            fi
        done
    fi

    if [ $type == "fcgf" ]; then
        for k in ${model_numbers[@]}
        do

            echo "model ${k}" >> ${file}
            folder=${criterion}_edge_filtering_pre_${preprocessing}_${type}_td_${training_data}_epoch_${epoch}
            rm -rf TestData/PartialNonDeformed/model${k}/${folder}
            mkdir TestData/PartialNonDeformed/model${k}/${folder}
            rm -rf TestData/PartialDeformed/model${k}/${folder}
            mkdir TestData/PartialDeformed/model${k}/${folder}

            if [ $criterion == "none" ]; then
                echo 'Partial Deformed' >> ${file}
                echo 'Edge filtering not used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialDeformed/model${k}/020_0.ply" \
                --t="PartialDeformed/model${k}/104_1.ply" \
                --s_feats="PartialDeformed/model${k}/020_0_fcgf.npz" \
                --t_feats="PartialDeformed/model${k}/104_1_fcgf.npz" \
                --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
                --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
                --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
                --output="PartialDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --preprocessing=${preprocessing} \
                --level=${n_deformed_levels} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialDeformed/model${k}/${folder}/result.ply" \
                    --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
                    --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
                    --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
                    --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                    --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                    --save_final_path="TestData/PartialDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi
            
                echo 'Partial Non Deformed' >> ${file}
                echo 'Edge filtering not used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                --t="PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                --s_feats="PartialNonDeformed/model${k}/mesh_transformed_0_fcgf.npz" \
                --t_feats="PartialNonDeformed/model${k}/mesh_transformed_1_fcgf.npz" \
                --source_trans="PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --target_trans="PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --matches="PartialNonDeformed/model${k}/0_1.npz" \
                --output="PartialNonDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialNonDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialNonDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --preprocessing=${preprocessing} \
                --level=${n_non_deformed_levels} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                    
                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialNonDeformed/model${k}/${folder}/result.ply" \
                    --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                    --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                    --save_final_path="TestData/PartialNonDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi

            else

                echo 'Partial Deformed' >> ${file}
                echo 'Edge filtering used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialDeformed/model${k}/020_0.ply" \
                --t="PartialDeformed/model${k}/104_1.ply" \
                --s_feats="PartialDeformed/model${k}/020_0_fcgf.npz" \
                --t_feats="PartialDeformed/model${k}/104_1_fcgf.npz" \
                --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
                --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
                --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
                --output="PartialDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --preprocessing=${preprocessing} \
                --confidence_threshold=${confidence_threshold} \
                --${edge_filtering_type} \
                --reject_outliers=false \
                --level=${n_deformed_levels} \
                --min_dist_thr=${min_dist_thr_fcgf} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                    
                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialDeformed/model${k}/${folder}/result.ply" \
                    --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
                    --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
                    --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
                    --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                    --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                    --save_final_path="TestData/PartialDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi

                echo 'Partial Non Deformed' >> ${file}
                echo 'Edge filtering used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                --t="PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                --s_feats="PartialNonDeformed/model${k}/mesh_transformed_0_fcgf.npz" \
                --t_feats="PartialNonDeformed/model${k}/mesh_transformed_1_fcgf.npz" \
                --source_trans="PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --target_trans="PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --matches="PartialNonDeformed/model${k}/0_1.npz" \
                --output="PartialNonDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialNonDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialNonDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --preprocessing=${preprocessing} \
                --confidence_threshold=${confidence_threshold} \
                --${edge_filtering_type} \
                --reject_outliers=false \
                --min_dist_thr=${min_dist_thr_fcgf} \
                --level=${n_non_deformed_levels} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                    
                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialNonDeformed/model${k}/${folder}/result.ply" \
                    --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                    --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                    --save_final_path="TestData/PartialNonDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi
            fi
        done
    fi
fi

if [ $current_deformation == "True" ]; then

    file="Testing/exterior_boundary_detection/testing_${criterion}_edge_filtering_pre_${preprocessing}_${type}_td_${training_data}_epoch_${epoch}_current_deformation.txt"
    rm ${file}
    touch ${file}

    if [ $type == "kpfcn" ]; then
        for k in ${model_numbers[@]}
        do
            echo "model ${k}" >> ${file}
            folder="${criterion}_edge_filtering_pre_${preprocessing}_${type}_td_${training_data}_epoch_${epoch}_current_deformation"
            rm -rf TestData/PartialNonDeformed/model${k}/${folder}
            mkdir TestData/PartialNonDeformed/model${k}/${folder}
            rm -rf TestData/PartialDeformed/model${k}/${folder}
            mkdir TestData/PartialDeformed/model${k}/${folder}

            if [ $criterion == "none" ]; then

                echo 'Partial Deformed' >> ${file}
                echo 'Edge filtering not used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialDeformed/model${k}/020_0.ply" \
                --t="PartialDeformed/model${k}/104_1.ply" \
                --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
                --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
                --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
                --output="PartialDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --preprocessing=${preprocessing} \
                --only_inference \
                --level=${n_deformed_levels} \
                --print_keypoints >> ${file}
                
                if [ "$?" != "1" ]; then

                    rm "${base}/PartialDeformed/model${k}/${folder}/current_deformation.ply"
                        
                    python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
                    --file1="${base}/PartialDeformed/model${k}/020_0.ply" \
                    --file2="${base}/PartialDeformed/model${k}/104_1.ply" \
                    --landmarks1="${base}/PartialDeformed/model${k}/${folder}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
                    --landmarks2="${base}/PartialDeformed/model${k}/${folder}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
                    --save_path="${base}/PartialDeformed/model${k}/${folder}/current_deformation.ply" >> ${file}

                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialDeformed/model${k}/${folder}/current_deformation.ply" \
                    --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
                    --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
                    --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
                    --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                    --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                    --save_final_path="TestData/PartialDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialDeformed/model${k}/${folder}/destination.ply" >> ${file}
                
                fi

                echo 'Partial Non Deformed' >> ${file}
                echo 'Edge filtering not used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                --t="PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                --source_trans="PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --target_trans="PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --matches="PartialNonDeformed/model${k}/0_1.npz" \
                --output="PartialNonDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialNonDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialNonDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --preprocessing=${preprocessing} \
                --only_inference \
                --level=${n_non_deformed_levels} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then

                    rm "${base}/PartialNonDeformed/model${k}/${folder}/current_deformation.ply"

                    python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
                    --file1="${base}/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --file2="${base}/PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                    --landmarks1="${base}/PartialNonDeformed/model${k}/${folder}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
                    --landmarks2="${base}/PartialNonDeformed/model${k}/${folder}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
                    --save_path="${base}/PartialNonDeformed/model${k}/${folder}/current_deformation.ply" >> ${file}

                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialNonDeformed/model${k}/${folder}/current_deformation.ply" \
                    --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                    --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                    --save_final_path="TestData/PartialNonDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder}/destination.ply" >> ${file}
                
                fi
                
            else

                echo 'Partial Deformed' >> ${file}
                echo 'Edge filtering used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialDeformed/model${k}/020_0.ply" \
                --t="PartialDeformed/model${k}/104_1.ply" \
                --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
                --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
                --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
                --output="PartialDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --${edge_filtering_type} \
                --reject_outliers=false \
                --min_dist_thr=${min_dist_thr_kpfcn} \
                --preprocessing=${preprocessing} \
                --only_inference \
                --level=${n_deformed_levels} \
                --print_keypoints >> ${file}
                
                if [ "$?" != "1" ]; then

                    rm "${base}/PartialDeformed/model${k}/${folder}/current_deformation.ply"
                            
                    python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
                    --file1="${base}/PartialDeformed/model${k}/020_0.ply" \
                    --file2="${base}/PartialDeformed/model${k}/104_1.ply" \
                    --landmarks1="${base}/PartialDeformed/model${k}/${folder}/${type}_${criterion}_edge_filtering_ldmk/${criterion}_edge_filtered_ldmk_s_pcd.ply" \
                    --landmarks2="${base}/PartialDeformed/model${k}/${folder}/${type}_${criterion}_edge_filtering_ldmk/${criterion}_edge_filtered_ldmk_t_pcd.ply" \
                    --save_path="${base}/PartialDeformed/model${k}/${folder}/current_deformation.ply" >> ${file}

                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialDeformed/model${k}/${folder}/current_deformation.ply" \
                    --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
                    --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
                    --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
                    --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                    --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                    --save_final_path="TestData/PartialDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi

                echo 'Partial Non Deformed' >> ${file}
                echo 'Edge filtering used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                --t="PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                --source_trans="PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --target_trans="PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --matches="PartialNonDeformed/model${k}/0_1.npz" \
                --output="PartialNonDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialNonDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialNonDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --${edge_filtering_type} \
                --reject_outliers=false \
                --min_dist_thr=${min_dist_thr_kpfcn} \
                --preprocessing=${preprocessing} \
                --only_inference \
                --level=${n_non_deformed_levels} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then

                    rm "${base}/PartialNonDeformed/model${k}/${folder}/current_deformation.ply"

                    python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
                    --file1="${base}/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --file2="${base}/PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                    --landmarks1="${base}/PartialNonDeformed/model${k}/${folder}/${type}_${criterion}_edge_filtering_ldmk/${criterion}_edge_filtered_ldmk_s_pcd.ply" \
                    --landmarks2="${base}/PartialNonDeformed/model${k}/${folder}/${type}_${criterion}_edge_filtering_ldmk/${criterion}_edge_filtered_ldmk_t_pcd.ply" \
                    --save_path="${base}/PartialNonDeformed/model${k}/${folder}/current_deformation.ply" >> ${file}

                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialNonDeformed/model${k}/${folder}/current_deformation.ply" \
                    --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                    --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                    --save_final_path="TestData/PartialNonDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi
            fi
        done
    fi

    if [ $type == "fcgf" ]; then
        for k in ${model_numbers[@]}
        do

            echo "model ${k}" >> ${file}
            folder=${criterion}_edge_filtering_pre_${preprocessing}_${type}_td_${training_data}_epoch_${epoch}
            rm -rf TestData/FullNonDeformed/model${k}/${folder}
            mkdir TestData/FullNonDeformed/model${k}/${folder}
            rm -rf TestData/FullDeformed/model${k}/${folder}
            mkdir TestData/FullDeformed/model${k}/${folder}
            rm -rf TestData/PartialNonDeformed/model${k}/${folder}
            mkdir TestData/PartialNonDeformed/model${k}/${folder}
            rm -rf TestData/PartialDeformed/model${k}/${folder}
            mkdir TestData/PartialDeformed/model${k}/${folder}

            if [ $criterion == "none" ]; then

                echo 'Partial Deformed' >> ${file}
                echo 'Edge filtering not used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialDeformed/model${k}/020_0.ply" \
                --t="PartialDeformed/model${k}/104_1.ply" \
                --s_feats="PartialDeformed/model${k}/020_0_fcgf.npz" \
                --t_feats="PartialDeformed/model${k}/104_1_fcgf.npz" \
                --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
                --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
                --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
                --output="PartialDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --preprocessing=${preprocessing} \
                --only_inference \
                --level=${n_deformed_levels} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then

                    rm "${base}/PartialDeformed/model${k}/${folder}/current_deformation.ply"
                            
                    python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
                    --file1="${base}/PartialDeformed/model${k}/020_0.ply" \
                    --file2="${base}/PartialDeformed/model${k}/104_1.ply" \
                    --landmarks1="${base}/PartialDeformed/model${k}/${folder}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
                    --landmarks2="${base}/PartialDeformed/model${k}/${folder}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
                    --save_path="${base}/PartialDeformed/model${k}/${folder}/current_deformation.ply" >> ${file}

                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialDeformed/model${k}/${folder}/current_deformation.ply" \
                    --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
                    --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
                    --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
                    --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                    --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                    --save_final_path="TestData/PartialDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi
            
                echo 'Partial Non Deformed' >> ${file}
                echo 'Edge filtering not used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                --t="PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                --s_feats="PartialNonDeformed/model${k}/mesh_transformed_0_fcgf.npz" \
                --t_feats="PartialNonDeformed/model${k}/mesh_transformed_1_fcgf.npz" \
                --source_trans="PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --target_trans="PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --matches="PartialNonDeformed/model${k}/0_1.npz" \
                --output="PartialNonDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialNonDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialNonDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --preprocessing=${preprocessing} \
                --only_inference \
                --level=${n_non_deformed_levels} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then

                    rm "${base}/PartialNonDeformed/model${k}/${folder}/current_deformation.ply"

                    python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
                    --file1="${base}/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --file2="${base}/PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                    --landmarks1="${base}/PartialNonDeformed/model${k}/${folder}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
                    --landmarks2="${base}/PartialNonDeformed/model${k}/${folder}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
                    --save_path="${base}/PartialNonDeformed/model${k}/${folder}/current_deformation.ply" >> ${file}

                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialNonDeformed/model${k}/${folder}/current_deformation.ply" \
                    --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                    --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                    --save_final_path="TestData/PartialNonDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi

            else

                echo 'Partial Deformed' >> ${file}
                echo 'Edge filtering used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialDeformed/model${k}/020_0.ply" \
                --t="PartialDeformed/model${k}/104_1.ply" \
                --s_feats="PartialDeformed/model${k}/020_0_fcgf.npz" \
                --t_feats="PartialDeformed/model${k}/104_1_fcgf.npz" \
                --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
                --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
                --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
                --output="PartialDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --preprocessing=${preprocessing} \
                --confidence_threshold=${confidence_threshold} \
                --${edge_filtering_type} \
                --reject_outliers=false \
                --min_dist_thr=${min_dist_thr_fcgf} \
                --only_inference \
                --level=${n_deformed_levels} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                
                    rm "${base}/PartialNonDeformed/model${k}/${folder}/current_deformation.ply"
                            
                    python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
                    --file1="${base}/PartialDeformed/model${k}/020_0.ply" \
                    --file2="${base}/PartialDeformed/model${k}/104_1.ply" \
                    --landmarks1="${base}/PartialDeformed/model${k}/${folder}/${type}_${criterion}_edge_filtering_ldmk/${criterion}_edge_filtered_ldmk_s_pcd.ply" \
                    --landmarks2="${base}/PartialDeformed/model${k}/${folder}/${type}_${criterion}_edge_filtering_ldmk/${criterion}_edge_filtered_ldmk_t_pcd.ply" \
                    --save_path="${base}/PartialDeformed/model${k}/${folder}/current_deformation.ply" >> ${file}

                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialDeformed/model${k}/${folder}/current_deformation.ply" \
                    --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
                    --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
                    --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
                    --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                    --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                    --save_final_path="TestData/PartialDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi

                echo 'Partial Non Deformed' >> ${file}
                echo 'Edge filtering used' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                --t="PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                --s_feats="PartialNonDeformed/model${k}/mesh_transformed_0_fcgf.npz" \
                --t_feats="PartialNonDeformed/model${k}/mesh_transformed_1_fcgf.npz" \
                --source_trans="PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --target_trans="PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --matches="PartialNonDeformed/model${k}/0_1.npz" \
                --output="PartialNonDeformed/model${k}/${folder}/result.ply" \
                --output_trans="PartialNonDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="PartialNonDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --preprocessing=${preprocessing} \
                --confidence_threshold=${confidence_threshold} \
                --${edge_filtering_type} \
                --reject_outliers=false \
                --min_dist_thr=${min_dist_thr_fcgf} \
                --only_inference \
                --level=${n_non_deformed_levels} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then

                    rm "${base}/PartialNonDeformed/model${k}/${folder}/current_deformation.ply"

                    python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
                    --file1="${base}/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --file2="${base}/PartialNonDeformed/model${k}/mesh_transformed_1.ply" \
                    --landmarks1="${base}/PartialNonDeformed/model${k}/${folder}/${type}_${criterion}_edge_filtering_ldmk/${criterion}_edge_filtered_ldmk_s_pcd.ply" \
                    --landmarks2="${base}/PartialNonDeformed/model${k}/${folder}/${type}_${criterion}_edge_filtering_ldmk/${criterion}_edge_filtered_ldmk_t_pcd.ply" \
                    --save_path="${base}/PartialNonDeformed/model${k}/${folder}/current_deformation.ply" >> ${file}

                    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                    --final="TestData/PartialNonDeformed/model${k}/${folder}/current_deformation.ply" \
                    --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                    --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                    --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                    --save_final_path="TestData/PartialNonDeformed/model${k}/${folder}/final.ply" \
                    --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi
            fi
        done
    fi
fi