type=fcgf
# type=kpfcn

td=full_deformed
# td=partial_deformed
# td=pretrained

if [ "$type" == "kpfcn" ] ; then
	config=LNDP.yaml
else
	config=LNDP_fcgf.yaml
fi

n_deformed_levels=8

preprocessing=mutual
preprocessing_default=mutual

confidence=0.000001
confidence_default=0.000001

number_centers=(5 10 20 50 100 150 200 300 500)

average_distance_multipliers=(3.0)

inlier_outlier_thrs=(0.01)

sampling=linspace

model_numbers=('002' '042' '085' '126' '167' '207')

# max_ldmks=50
max_ldmks=None

if [ $type == "kpfcn" ] 
then
    for k in ${model_numbers[@]}
    do

        # Using Lepard matching

        folder_name=output_lepard_default_${type}_td_${td}
        file_name=Testing/custom_filtering/output_lepard_default_${type}_td_${td}_model_${k}.txt

        rm -rf TestData/FullNonDeformed/model${k}/${folder_name}
        mkdir TestData/FullNonDeformed/model${k}/${folder_name}
        rm -rf TestData/FullDeformed/model${k}/${folder_name}
        mkdir TestData/FullDeformed/model${k}/${folder_name}
        rm -rf TestData/PartialNonDeformed/model${k}/${folder_name}
        mkdir TestData/PartialNonDeformed/model${k}/${folder_name}
        rm -rf TestData/PartialDeformed/model${k}/${folder_name}
        mkdir TestData/PartialDeformed/model${k}/${folder_name}

        rm ${file_name}
        touch ${file_name}

        echo "Lepard" >> ${file_name}
        echo "model ${k}" >> ${file_name}

        if [ $td == "partial_deformed" ] 
        then

            echo "Partial Deformed" >> ${file_name} 
            CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
            --config=config/${config} \
            --s="PartialDeformed/model${k}/020_0.ply" \
            --t="PartialDeformed/model${k}/104_1.ply" \
            --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
            --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
            --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
            --mesh_path="PartialDeformed/model${k}/mesh_020_0.ply" \
            --output="PartialDeformed/model${k}/${folder_name}/result.ply" \
            --output_trans="PartialDeformed/model${k}/${folder_name}/result_se4.h5" \
            --intermediate_output_folder="PartialDeformed/model${k}/${folder_name}/" \
            --base="/home/aiday.kyzy/code/DeformationPyramid/TestData/" \
            --print_keypoints \
            --confidence_threshold=${confidence_default} \
            --level=${n_deformed_levels} \
            --preprocessing=${preprocessing_default} \
            --reject_outliers=false >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
            --pred="TestData/PartialDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}
            
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --final="TestData/PartialDeformed/model${k}/${folder_name}/result.ply" \
            --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
            --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
            --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
            --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
            --save_final_path="TestData/PartialDeformed/model${k}/${folder_name}/final.ply" \
            --save_destination_path="TestData/PartialDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}
        fi

        if [ $td == "full_deformed" ] || [ $td == "pretrained" ] 
        then
            echo "Full Deformed" >> ${file_name} 
            CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
            --config=config/${config} \
            --s="FullDeformed/model${k}/020.ply" \
            --t="FullDeformed/model${k}/104.ply" \
            --source_trans="FullDeformed/model${k}/020_se4.h5" \
            --target_trans="FullDeformed/model${k}/104_se4.h5" \
            --matches="FullDeformed/model${k}/020_104.npz" \
            --mesh_path="FullDeformed/model${k}/mesh_020.ply" \
            --output="FullDeformed/model${k}/${folder_name}/result.ply" \
            --output_trans="FullDeformed/model${k}/${folder_name}/result_se4.h5" \
            --intermediate_output_folder="FullDeformed/model${k}/${folder_name}/" \
            --base="/home/aiday.kyzy/code/DeformationPyramid/TestData/" \
            --print_keypoints  \
            --confidence_threshold=${confidence_default} \
            --level=${n_deformed_levels} \
            --preprocessing=${preprocessing_default} \
            --reject_outliers=false >> ${file_name}
            
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            --part1="TestData/FullDeformed/model${k}/020_se4.h5" \
            --part2="TestData/FullDeformed/model${k}/104_se4.h5" \
            --pred="TestData/FullDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}
            
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --input1="TestData/FullDeformed/model${k}/${folder_name}/result.ply" \
            --input2="TestData/FullDeformed/model${k}/104.ply" \
            --matches="TestData/FullDeformed/model${k}/020_104.npz" \
            --save_final_path="TestData/FullDeformed/model${k}/${folder_name}/final.ply" \
            --save_destination_path="TestData/FullDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}
        fi

        # Using Outlier Rejection

        folder_name=output_outlier_rejection_default_${type}_td_${td}
        file_name=Testing/custom_filtering/output_outlier_rejection_default_${type}_td_${td}_model_${k}.txt

        rm -rf TestData/FullNonDeformed/model${k}/${folder_name}
        mkdir TestData/FullNonDeformed/model${k}/${folder_name}
        rm -rf TestData/FullDeformed/model${k}/${folder_name}
        mkdir TestData/FullDeformed/model${k}/${folder_name}
        rm -rf TestData/PartialNonDeformed/model${k}/${folder_name}
        mkdir TestData/PartialNonDeformed/model${k}/${folder_name}
        rm -rf TestData/PartialDeformed/model${k}/${folder_name}
        mkdir TestData/PartialDeformed/model${k}/${folder_name}

        rm ${file_name}
        touch ${file_name}

        echo "Outlier Rejection" >> ${file_name}
        echo "model ${k}" >> ${file_name}

        if [ $td == "partial_deformed" ] 
        then
            echo "Partial Deformed" >> ${file_name} 
            CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
            --config=config/${config} \
            --s="PartialDeformed/model${k}/020_0.ply" \
            --t="PartialDeformed/model${k}/104_1.ply" \
            --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
            --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
            --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
            --mesh_path="PartialDeformed/model${k}/mesh_020_0.ply" \
            --output="PartialDeformed/model${k}/${folder_name}/result.ply" \
            --output_trans="PartialDeformed/model${k}/${folder_name}/result_se4.h5" \
            --intermediate_output_folder="PartialDeformed/model${k}/${folder_name}/" \
            --base="/home/aiday.kyzy/code/DeformationPyramid/TestData/" \
            --print_keypoints  \
            --confidence_threshold=${confidence_default} \
            --level=${n_deformed_levels} \
            --preprocessing=${preprocessing_default} \
            --reject_outliers=true  >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
            --pred="TestData/PartialDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}
            
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --final="TestData/PartialDeformed/model${k}/${folder_name}/result.ply" \
            --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
            --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
            --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
            --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
            --save_final_path="TestData/PartialDeformed/model${k}/${folder_name}/final.ply" \
            --save_destination_path="TestData/PartialDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}
        fi
        
        if [ $td == "full_deformed" ] || [ $td == "pretrained" ] 
        then
            echo "Full Deformed" >> ${file_name} 
            CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
            --config=config/${config} \
            --s="FullDeformed/model${k}/020.ply" \
            --t="FullDeformed/model${k}/104.ply" \
            --source_trans="FullDeformed/model${k}/020_se4.h5" \
            --target_trans="FullDeformed/model${k}/104_se4.h5" \
            --matches="FullDeformed/model${k}/020_104.npz" \
            --mesh_path="FullDeformed/model${k}/mesh_020.ply" \
            --output="FullDeformed/model${k}/${folder_name}/result.ply" \
            --output_trans="FullDeformed/model${k}/${folder_name}/result_se4.h5" \
            --intermediate_output_folder="FullDeformed/model${k}/${folder_name}/" \
            --base="/home/aiday.kyzy/code/DeformationPyramid/TestData/" \
            --print_keypoints  \
            --confidence_threshold=${confidence_default} \
            --level=${n_deformed_levels} \
            --preprocessing=${preprocessing_default} \
            --reject_outliers=true >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            --part1="TestData/FullDeformed/model${k}/020_se4.h5" \
            --part2="TestData/FullDeformed/model${k}/104_se4.h5" \
            --pred="TestData/FullDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --input1="TestData/FullDeformed/model${k}/${folder_name}/result.ply" \
            --input2="TestData/FullDeformed/model${k}/104.ply" \
            --matches="TestData/FullDeformed/model${k}/020_104.npz" \
            --save_final_path="TestData/FullDeformed/model${k}/${folder_name}/final.ply" \
            --save_destination_path="TestData/FullDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}
        fi

        ### Using Custom Filtering

        for number_center in "${number_centers[@]}"
        do
            for average_distance_multiplier in "${average_distance_multipliers[@]}"
            do
                for inlier_outlier_thr in "${inlier_outlier_thrs[@]}"
                do
                    echo "model ${k}" >> ${file_name}
                    echo "number of centers ${number_center}" >> ${file_name}
                    echo "average distance multiplier ${average_distance_multiplier}" >> ${file_name}
                    echo "inlier outlier thresholds ${inlier_outlier_thr}" >> ${file_name}

                    folder_name=output_final_p_${preprocessing}_c_${confidence}_nc_${number_center}_adm_${average_distance_multiplier}_iot_${inlier_outlier_thr}_s_${sampling}_max_ldmks_${max_ldmks}_${type}_td_${td}
                    file_name=Testing/custom_filtering/p_${preprocessing}_c_${confidence}_nc_${number_center}_adm_${average_distance_multiplier}_iot_${inlier_outlier_thr}_s_${sampling}_max_ldmks_${max_ldmks}_${type}_td_${td}_model_${k}.txt

                    rm -rf TestData/FullNonDeformed/model${k}/${folder_name}
                    mkdir TestData/FullNonDeformed/model${k}/${folder_name}
                    rm -rf TestData/FullDeformed/model${k}/${folder_name}
                    mkdir TestData/FullDeformed/model${k}/${folder_name}
                    rm -rf TestData/PartialNonDeformed/model${k}/${folder_name}
                    mkdir TestData/PartialNonDeformed/model${k}/${folder_name}
                    rm -rf TestData/PartialDeformed/model${k}/${folder_name}
                    mkdir TestData/PartialDeformed/model${k}/${folder_name}

                    rm ${file_name}
                    touch ${file_name}

                    if [ $td == "partial_deformed" ] 
                    then

                        echo "Partial Deformed" >> ${file_name} 
                        CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                        --config=config/${config} \
                        --s="PartialDeformed/model${k}/020_0.ply" \
                        --t="PartialDeformed/model${k}/104_1.ply" \
                        --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
                        --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
                        --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
                        --mesh_path="PartialDeformed/model${k}/mesh_020_0.ply" \
                        --output="PartialDeformed/model${k}/${folder_name}/result.ply" \
                        --output_trans="PartialDeformed/model${k}/${folder_name}/result_se4.h5" \
                        --intermediate_output_folder="PartialDeformed/model${k}/${folder_name}/" \
                        --base="/home/aiday.kyzy/code/DeformationPyramid/TestData/" \
                        --confidence_threshold=${confidence} \
                        --preprocessing=${preprocessing} \
                        --index_coarse_feats=1 \
                        --coarse_level=${coarse_level} \
                        --number_centers=${number_center} \
                        --average_distance_multiplier=${average_distance_multiplier} \
                        --number_iterations_custom_filtering=1 \
                        --inlier_outlier_thr=${inlier_outlier_thr} \
                        --sampling=${sampling} \
                        --max_ldmks=${max_ldmks} \
                        --print_keypoints \
                        --level=${n_deformed_levels} \
                        --custom_filtering  >> ${file_name}

                        python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                        --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                        --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                        --pred="TestData/PartialDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}

                        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                        --final="TestData/PartialDeformed/model${k}/${folder_name}/result.ply" \
                        --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
                        --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
                        --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
                        --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                        --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                        --save_final_path="TestData/PartialDeformed/model${k}/${folder_name}/final.ply" \
                        --save_destination_path="TestData/PartialDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}
                    fi

                    if [ $td == "full_deformed" ] || [ $td == "pretrained" ] 
                    then
                        echo "Full Deformed" >> ${file_name} 
                        CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                        --config=config/${config} \
                        --s="FullDeformed/model${k}/020.ply" \
                        --t="FullDeformed/model${k}/104.ply" \
                        --source_trans="FullDeformed/model${k}/020_se4.h5" \
                        --target_trans="FullDeformed/model${k}/104_se4.h5" \
                        --matches="FullDeformed/model${k}/020_104.npz" \
                        --mesh_path="FullDeformed/model${k}/mesh_020.ply" \
                        --output="FullDeformed/model${k}/${folder_name}/result.ply" \
                        --output_trans="FullDeformed/model${k}/${folder_name}/result_se4.h5" \
                        --intermediate_output_folder="FullDeformed/model${k}/${folder_name}/" \
                        --base="/home/aiday.kyzy/code/DeformationPyramid/TestData/" \
                        --confidence_threshold=${confidence} \
                        --preprocessing=${preprocessing} \
                        --index_coarse_feats=1 \
                        --coarse_level=${coarse_level} \
                        --number_centers=${number_center} \
                        --average_distance_multiplier=${average_distance_multiplier} \
                        --number_iterations_custom_filtering=1 \
                        --inlier_outlier_thr=${inlier_outlier_thr} \
                        --sampling=${sampling} \
                        --max_ldmks=${max_ldmks} \
                        --print_keypoints \
                        --level=${n_deformed_levels} \
                        --custom_filtering   >> ${file_name}

                        python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                        --part1="TestData/FullDeformed/model${k}/020_se4.h5" \
                        --part2="TestData/FullDeformed/model${k}/104_se4.h5" \
                        --pred="TestData/FullDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}

                        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                        --input1="TestData/FullDeformed/model${k}/${folder_name}/result.ply" \
                        --input2="TestData/FullDeformed/model${k}/104.ply" \
                        --matches="TestData/FullDeformed/model${k}/020_104.npz" \
                        --save_final_path="TestData/FullDeformed/model${k}/${folder_name}/final.ply" \
                        --save_destination_path="TestData/FullDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}
                    fi
                done
            done
        done
    done
fi

if [ $type == "fcgf" ]; then
    for k in ${model_numbers[@]}
    do
        
        # Using Lepard matching

        folder_name=output_lepard_default_${type}_td_${td}
        file_name=Testing/custom_filtering/output_lepard_default_${type}_td_${td}_model_${k}.txt

        rm -rf TestData/FullNonDeformed/model${k}/${folder_name}
        mkdir TestData/FullNonDeformed/model${k}/${folder_name}
        rm -rf TestData/FullDeformed/model${k}/${folder_name}
        mkdir TestData/FullDeformed/model${k}/${folder_name}
        rm -rf TestData/PartialNonDeformed/model${k}/${folder_name}
        mkdir TestData/PartialNonDeformed/model${k}/${folder_name}
        rm -rf TestData/PartialDeformed/model${k}/${folder_name}
        mkdir TestData/PartialDeformed/model${k}/${folder_name}

        rm ${file_name}
        touch ${file_name}
        
        echo "Lepard" >> ${file_name}
        echo "model ${k}" >> ${file_name}

        if [ $td == "partial_deformed" ] 
        then

            echo 'Partial Deformed' >> ${file_name} 
            CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
            --config=config/${config} \
            --s="PartialDeformed/model${k}/020_0.ply" \
            --t="PartialDeformed/model${k}/104_1.ply" \
            --s_feats="PartialDeformed/model${k}/020_0_fcgf.npz" \
            --t_feats="PartialDeformed/model${k}/104_1_fcgf.npz" \
            --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
            --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
            --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
            --mesh_path="PartialDeformed/model${k}/mesh_020_0.ply" \
            --output="PartialDeformed/model${k}/${folder_name}/result.ply" \
            --output_trans="PartialDeformed/model${k}/${folder_name}/result_se4.h5" \
            --intermediate_output_folder="PartialDeformed/model${k}/${folder_name}/" \
            --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
            --print_keypoints  \
            --reject_outliers=false \
            --confidence_threshold=${confidence_default} \
            --level=${n_deformed_levels} \
            --preprocessing=${preprocessing_default} >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
            --pred="TestData/PartialDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --final="TestData/PartialDeformed/model${k}/${folder_name}/result.ply" \
            --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
            --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
            --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
            --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
            --save_final_path="TestData/PartialDeformed/model${k}/${folder_name}/final.ply" \
            --save_destination_path="TestData/PartialDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}
        fi

        if [ $td == "full_deformed" ] || [ $td == "pretrained" ] 
        then
            
            echo 'Full Deformed' >> ${file_name} 
            CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
            --config=config/${config} \
            --s="FullDeformed/model${k}/020.ply" \
            --t="FullDeformed/model${k}/104.ply" \
            --s_feats="FullDeformed/model${k}/020_fcgf.npz" \
            --t_feats="FullDeformed/model${k}/104_fcgf.npz" \
            --source_trans="FullDeformed/model${k}/020_se4.h5" \
            --target_trans="FullDeformed/model${k}/104_se4.h5" \
            --matches="FullDeformed/model${k}/020_104.npz" \
            --mesh_path="FullDeformed/model${k}/mesh_020.ply" \
            --output="FullDeformed/model${k}/${folder_name}/result.ply" \
            --output_trans="FullDeformed/model${k}/${folder_name}/result_se4.h5" \
            --intermediate_output_folder="FullDeformed/model${k}/${folder_name}/" \
            --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
            --print_keypoints  \
            --reject_outliers=false \
            --confidence_threshold=${confidence_default} \
            --level=${n_deformed_levels} \
            --preprocessing=${preprocessing_default} >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            --part1="TestData/FullDeformed/model${k}/020_se4.h5" \
            --part2="TestData/FullDeformed/model${k}/104_se4.h5" \
            --pred="TestData/FullDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --input1="TestData/FullDeformed/model${k}/${folder_name}/result.ply" \
            --input2="TestData/FullDeformed/model${k}/104.ply" \
            --matches="TestData/FullDeformed/model${k}/020_104.npz" \
            --save_final_path="TestData/FullDeformed/model${k}/${folder_name}/final.ply" \
            --save_destination_path="TestData/FullDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}
        fi

        # Using Outlier Rejection

        folder_name=output_outlier_rejection_default_${type}_td_${td}
        file_name=Testing/custom_filtering/output_outlier_rejection_default_${type}_td_${td}_model_${k}.txt

        rm -rf TestData/FullNonDeformed/model${k}/${folder_name}
        mkdir TestData/FullNonDeformed/model${k}/${folder_name}
        rm -rf TestData/FullDeformed/model${k}/${folder_name}
        mkdir TestData/FullDeformed/model${k}/${folder_name}
        rm -rf TestData/PartialNonDeformed/model${k}/${folder_name}
        mkdir TestData/PartialNonDeformed/model${k}/${folder_name}
        rm -rf TestData/PartialDeformed/model${k}/${folder_name}
        mkdir TestData/PartialDeformed/model${k}/${folder_name}

        rm ${file_name}
        touch ${file_name}

        echo 'Outlier Rejection' >> ${file_name}
        echo "model ${k}" >> ${file_name}

        if [ $td == "partial_deformed" ] 
        then

            echo 'Partial Deformed' >> ${file_name} 
            CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
            --config=config/${config} \
            --s="PartialDeformed/model${k}/020_0.ply" \
            --t="PartialDeformed/model${k}/104_1.ply" \
            --s_feats="PartialDeformed/model${k}/020_0_fcgf.npz" \
            --t_feats="PartialDeformed/model${k}/104_1_fcgf.npz" \
            --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
            --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
            --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
            --mesh_path="PartialDeformed/model${k}/mesh_020_0.ply" \
            --output="PartialDeformed/model${k}/${folder_name}/result.ply" \
            --output_trans="PartialDeformed/model${k}/${folder_name}/result_se4.h5" \
            --intermediate_output_folder="PartialDeformed/model${k}/${folder_name}/" \
            --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
            --print_keypoints  \
            --reject_outliers=true \
            --confidence_threshold=${confidence_default} \
            --level=${n_deformed_levels} \
            --preprocessing=${preprocessing_default} >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
            --pred="TestData/PartialDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --final="TestData/PartialDeformed/model${k}/${folder_name}/result.ply" \
            --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
            --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
            --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
            --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
            --save_final_path="TestData/PartialDeformed/model${k}/${folder_name}/final.ply" \
            --save_destination_path="TestData/PartialDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}
        fi

        if [ $td == "full_deformed" ] || [ $td == "pretrained" ] 
        then
            
            echo 'Full Deformed' >> ${file_name} 
            CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
            --config=config/${config} \
            --s="FullDeformed/model${k}/020.ply" \
            --t="FullDeformed/model${k}/104.ply" \
            --s_feats="FullDeformed/model${k}/020_fcgf.npz" \
            --t_feats="FullDeformed/model${k}/104_fcgf.npz" \
            --source_trans="FullDeformed/model${k}/020_se4.h5" \
            --target_trans="FullDeformed/model${k}/104_se4.h5" \
            --matches="FullDeformed/model${k}/020_104.npz" \
            --mesh_path="FullDeformed/model${k}/mesh_020.ply" \
            --output="FullDeformed/model${k}/${folder_name}/result.ply" \
            --output_trans="FullDeformed/model${k}/${folder_name}/result_se4.h5" \
            --intermediate_output_folder="FullDeformed/model${k}/${folder_name}/" \
            --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
            --print_keypoints  \
            --reject_outliers=true \
            --confidence_threshold=${confidence_default} \
            --level=${n_deformed_levels} \
            --preprocessing=${preprocessing_default} >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            --part1="TestData/FullDeformed/model${k}/020_se4.h5" \
            --part2="TestData/FullDeformed/model${k}/104_se4.h5" \
            --pred="TestData/FullDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}

            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --input1="TestData/FullDeformed/model${k}/${folder_name}/result.ply" \
            --input2="TestData/FullDeformed/model${k}/104.ply" \
            --matches="TestData/FullDeformed/model${k}/020_104.npz" \
            --save_final_path="TestData/FullDeformed/model${k}/${folder_name}/final.ply" \
            --save_destination_path="TestData/FullDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}

        fi

        # Using Custom Filtering

        for number_center in "${number_centers[@]}"
        do
            for average_distance_multiplier in "${average_distance_multipliers[@]}"
            do
                for inlier_outlier_thr in "${inlier_outlier_thrs[@]}"
                do
                    echo "model ${k}" >> ${file_name}
                    echo "number of centers ${number_center}" >> ${file_name}
                    echo "average distance multiplier ${average_distance_multiplier}" >> ${file_name}
                    echo "inlier outlier thresholds ${inlier_outlier_thr}" >> ${file_name}

                    folder_name=output_final_p_${preprocessing}_c_${confidence}_nc_${number_center}_adm_${average_distance_multiplier}_iot_${inlier_outlier_thr}_s_${sampling}_max_ldmks_${max_ldmks}_${type}_td_${td}
                    file_name=Testing/custom_filtering/p_${preprocessing}_c_${confidence}_nc_${number_center}_adm_${average_distance_multiplier}_iot_${inlier_outlier_thr}_s_${sampling}_max_ldmks_${max_ldmks}_${type}_td_${td}_model_${k}.txt

                    rm -rf TestData/FullNonDeformed/model${k}/${folder_name}
                    mkdir TestData/FullNonDeformed/model${k}/${folder_name}
                    rm -rf TestData/FullDeformed/model${k}/${folder_name}
                    mkdir TestData/FullDeformed/model${k}/${folder_name}
                    rm -rf TestData/PartialNonDeformed/model${k}/${folder_name}
                    mkdir TestData/PartialNonDeformed/model${k}/${folder_name}
                    rm -rf TestData/PartialDeformed/model${k}/${folder_name}
                    mkdir TestData/PartialDeformed/model${k}/${folder_name}

                    rm ${file_name}
                    touch ${file_name}
                    
                    if [ $td == "partial_deformed" ] 
                    then

                        echo 'Partial Deformed' >> ${file_name} 
                        CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                        --config=config/${config} \
                        --s="PartialDeformed/model${k}/020_0.ply" \
                        --t="PartialDeformed/model${k}/104_1.ply" \
                        --s_feats="PartialDeformed/model${k}/020_0_fcgf.npz" \
                        --t_feats="PartialDeformed/model${k}/104_1_fcgf.npz" \
                        --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
                        --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
                        --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
                        --mesh_path="PartialDeformed/model${k}/mesh_020_0.ply" \
                        --output="PartialDeformed/model${k}/${folder_name}/result.ply" \
                        --output_trans="PartialDeformed/model${k}/${folder_name}/result_se4.h5" \
                        --intermediate_output_folder="PartialDeformed/model${k}/${folder_name}/" \
                        --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                        --confidence_threshold=${confidence} \
                        --preprocessing=${preprocessing} \
                        --index_coarse_feats=1 \
                        --coarse_level=${coarse_level} \
                        --number_centers=${number_center} \
                        --average_distance_multiplier=${average_distance_multiplier} \
                        --number_iterations_custom_filtering=1 \
                        --inlier_outlier_thr=${inlier_outlier_thr} \
                        --sampling=${sampling} \
                        --max_ldmks=${max_ldmks} \
                        --print_keypoints \
                        --custom_filtering  \
                        --level=${n_deformed_levels} \
                        --preprocessing=${preprocessing} >> ${file_name}

                        if [ "$?" != "1" ]; then
                        python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                        --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                        --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                        --pred="TestData/PartialDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}

                        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                        --final="TestData/PartialDeformed/model${k}/${folder_name}/result.ply" \
                        --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" \
                        --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" \
                        --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" \
                        --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                        --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                        --save_final_path="TestData/PartialDeformed/model${k}/${folder_name}/final.ply" \
                        --save_destination_path="TestData/PartialDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}
                        fi
                    fi

                    if [ $td == "full_deformed" ] || [ $td == "pretrained" ] 
                    then
                        echo 'Full Deformed' >> ${file_name} 
                        CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                        --config=config/${config} \
                        --s="FullDeformed/model${k}/020.ply" \
                        --t="FullDeformed/model${k}/104.ply" \
                        --s_feats="FullDeformed/model${k}/020_fcgf.npz" \
                        --t_feats="FullDeformed/model${k}/104_fcgf.npz"  \
                        --source_trans="FullDeformed/model${k}/020_se4.h5" \
                        --target_trans="FullDeformed/model${k}/104_se4.h5" \
                        --matches="FullDeformed/model${k}/020_104.npz" \
                        --mesh_path="FullDeformed/model${k}/mesh_020.ply" \
                        --output="FullDeformed/model${k}/${folder_name}/result.ply" \
                        --output_trans="FullDeformed/model${k}/${folder_name}/result_se4.h5" \
                        --intermediate_output_folder="FullDeformed/model${k}/${folder_name}/" \
                        --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                        --confidence_threshold=${confidence} \
                        --preprocessing=${preprocessing} \
                        --index_coarse_feats=1 \
                        --coarse_level=${coarse_level} \
                        --number_centers=${number_center} \
                        --average_distance_multiplier=${average_distance_multiplier} \
                        --number_iterations_custom_filtering=1 \
                        --inlier_outlier_thr=${inlier_outlier_thr} \
                        --sampling=${sampling} \
                        --max_ldmks=${max_ldmks} \
                        --print_keypoints \
                        --custom_filtering  \
                        --level=${n_deformed_levels} \
                        --preprocessing=${preprocessing} >> ${file_name}

                        if [ "$?" != "1" ]; then
                        python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                        --part1="TestData/FullDeformed/model${k}/020_se4.h5" \
                        --part2="TestData/FullDeformed/model${k}/104_se4.h5" \
                        --pred="TestData/FullDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}

                        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                        --input1="TestData/FullDeformed/model${k}/${folder_name}/result.ply" \
                        --input2="TestData/FullDeformed/model${k}/104.ply" \
                        --matches="TestData/FullDeformed/model${k}/020_104.npz" \
                        --save_final_path="TestData/FullDeformed/model${k}/${folder_name}/final.ply" \
                        --save_destination_path="TestData/FullDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}
                        fi
                    fi
                done
            done
        done
    done
fi