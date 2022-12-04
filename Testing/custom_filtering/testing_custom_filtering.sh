# version=1
# version=2
version=4

custom_filtering=true
# custom_filtering=false
if [ "$custom_filtering" = true ] ; then
    type=custom
else
    type=default
fi

preprocessing=none
# preprocessing=single
# preprocessing=mutual

confidence=0.1
# confidence=0.05
# confidence=0.01

coarse_level=-2
# coarse_level=-3

index_coarse_feats=1
# index_coarse_feats=2

# number_center=100
# number_center=200
# number_center=300
# number_center=400
# number_centers=(100 200 300)
number_centers=(1 5 10 50 100 200)

# average_distance_multiplier=1
# average_distance_multiplier=2
# average_distance_multiplier=3
# average_distance_multiplier=4
# average_distance_multipliers=(1 2 3 4)
# average_distance_multipliers=(1.0 1.4 1.8 2.2 2.6 3.0 3.4 3.8 4.2 4.6 5.0)
average_distance_multipliers=(3.0)

number_iterations_custom_filtering=1
# number_iterations_custom_filtering=2
# number_iterations_custom_filtering=3

# inlier_outlier_thrs=(0.005 0.01 0.02 0.03 0.04 0.05)
# inlier_outlier_thrs=(0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05)
inlier_outlier_thrs=(0.01)

sampling=linspace
# sampling=poisson

model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')

for k in ${model_numbers[@]}
do
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

            folder_name=output_v_${version}_t_${type}_p_${preprocessing}_c_${confidence}_nc_${number_center}_adm_${average_distance_multiplier}_cl_${coarse_level}_ic_${index_coarse_feats}_ni_${number_iterations_custom_filtering}_iot_${inlier_outlier_thr}_s_${sampling}
            file_name=Testing/custom_filtering/v_${version}_t_${type}_p_${preprocessing}_c_${confidence}_nc_${number_center}_adm_${average_distance_multiplier}_cl_${coarse_level}_ic_${index_coarse_feats}_ni_${number_iterations_custom_filtering}_iot_${inlier_outlier_thr}_s_${sampling}.txt

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

            if [ "$custom_filtering" = true ] ; then
                echo 'Full Non Deformed' >> ${file_name} 
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s="FullNonDeformed/model${k}/mesh_transformed_0.ply" --t="FullNonDeformed/model${k}/mesh_transformed_1.ply" --source_trans="FullNonDeformed/model${k}/mesh_transformed_0_se4.h5" --target_trans="FullNonDeformed/model${k}/mesh_transformed_1_se4.h5" --matches="FullNonDeformed/model${k}/0_1.npz" --mesh_path="FullNonDeformed/model${k}/mesh.ply" --output="FullNonDeformed/model${k}/${folder_name}/result.ply" --output_trans="FullNonDeformed/model${k}/${folder_name}/result_se4.h5" --intermediate_output_folder="FullNonDeformed/model${k}/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_center} --average_distance_multiplier=${average_distance_multiplier} --number_iterations_custom_filtering=1 --inlier_outlier_thr=${inlier_outlier_thr} --sampling=${sampling} --print_keypoints --custom_filtering --show_lepard_inliers >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/FullNonDeformed/model${k}/mesh_transformed_0_se4.h5" --part2="TestData/FullNonDeformed/model${k}/mesh_transformed_1_se4.h5" --pred="TestData/FullNonDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullNonDeformed/model${k}/${folder_name}/result.ply" --input2="TestData/FullNonDeformed/model${k}/mesh_transformed_1.ply" --save_final_path="TestData/FullNonDeformed/model${k}/${folder_name}/final.ply" --save_destination_path="TestData/FullNonDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}

                echo 'Partial Deformed' >> ${file_name} 
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s="PartialDeformed/model${k}/020_0.ply" --t="PartialDeformed/model${k}/104_1.ply" --source_trans="PartialDeformed/model${k}/020_0_se4.h5" --target_trans="PartialDeformed/model${k}/104_1_se4.h5" --matches="PartialDeformed/model${k}/020_104_0_1.npz" --mesh_path="PartialDeformed/model${k}/mesh_020_0.ply" --output="PartialDeformed/model${k}/${folder_name}/result.ply" --output_trans="PartialDeformed/model${k}/${folder_name}/result_se4.h5" --intermediate_output_folder="PartialDeformed/model${k}/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_center} --average_distance_multiplier=${average_distance_multiplier} --number_iterations_custom_filtering=1 --inlier_outlier_thr=${inlier_outlier_thr} --sampling=${sampling} --print_keypoints --custom_filtering --show_lepard_inliers >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" --pred="TestData/PartialDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="TestData/PartialDeformed/model${k}/${folder_name}/result.ply" --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" --save_final_path="TestData/PartialDeformed/model${k}/${folder_name}/final.ply" --save_destination_path="TestData/PartialDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}

                echo 'Full Deformed' >> ${file_name} 
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s="FullDeformed/model${k}/020.ply" --t="FullDeformed/model${k}/104.ply" --source_trans="FullDeformed/model${k}/020_se4.h5" --target_trans="FullDeformed/model${k}/104_se4.h5" --matches="FullDeformed/model${k}/020_104.npz" --mesh_path="FullDeformed/model${k}/mesh_020.ply" --output="FullDeformed/model${k}/${folder_name}/result.ply" --output_trans="FullDeformed/model${k}/${folder_name}/result_se4.h5" --intermediate_output_folder="FullDeformed/model${k}/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_center} --average_distance_multiplier=${average_distance_multiplier} --number_iterations_custom_filtering=1 --inlier_outlier_thr=${inlier_outlier_thr} --sampling=${sampling} --print_keypoints --custom_filtering --show_lepard_inliers  >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/FullDeformed/model${k}/020_se4.h5" --part2="TestData/FullDeformed/model${k}/104_se4.h5" --pred="TestData/FullDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullDeformed/model${k}/${folder_name}/result.ply" --input2="TestData/FullDeformed/model${k}/104.ply" --matches="TestData/FullDeformed/model${k}/020_104.npz" --save_final_path="TestData/FullDeformed/model${k}/${folder_name}/final.ply" --save_destination_path="TestData/FullDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}

                echo 'Partial Non Deformed' >> ${file_name} 
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s="PartialNonDeformed/model${k}/mesh_transformed_0.ply" --t="PartialNonDeformed/model${k}/mesh_transformed_1.ply" --source_trans="PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" --target_trans="PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" --matches="PartialNonDeformed/model${k}/0_1.npz" --mesh_path="PartialNonDeformed/model${k}/mesh_0.ply" --output="PartialNonDeformed/model${k}/${folder_name}/result.ply" --output_trans="PartialNonDeformed/model${k}/${folder_name}/result_se4.h5" --intermediate_output_folder="PartialNonDeformed/model${k}/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_center} --average_distance_multiplier=${average_distance_multiplier} --number_iterations_custom_filtering=1 --inlier_outlier_thr=${inlier_outlier_thr} --sampling=${sampling} --print_keypoints --custom_filtering --show_lepard_inliers >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" --pred="TestData/PartialNonDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="TestData/PartialNonDeformed/model${k}/${folder_name}/result.ply" --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" --save_final_path="TestData/PartialNonDeformed/model${k}/${folder_name}/final.ply" --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}

            else
                echo 'Full Non Deformed' >> ${file_name} 
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s="FullNonDeformed/model${k}/mesh_transformed_0.ply" --t="FullNonDeformed/model${k}/mesh_transformed_1.ply" --source_trans="FullNonDeformed/model${k}/mesh_transformed_0_se4.h5" --target_trans="FullNonDeformed/model${k}/mesh_transformed_1_se4.h5" --matches="FullNonDeformed/model${k}/0_1.npz" --mesh_path="FullNonDeformed/model${k}/mesh.ply" --output="FullNonDeformed/model${k}/${folder_name}/result.ply" --output_trans="FullNonDeformed/model${k}/${folder_name}/result_se4.h5" --intermediate_output_folder="FullNonDeformed/model${k}/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_center} --average_distance_multiplier=${average_distance_multiplier} --number_iterations_custom_filtering=1 --inlier_outlier_thr=${inlier_outlier_thr} --sampling=${sampling} --print_keypoints --show_lepard_inliers >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/FullNonDeformed/model${k}/mesh_transformed_0_se4.h5" --part2="TestData/FullNonDeformed/model${k}/mesh_transformed_1_se4.h5" --pred="TestData/FullNonDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullNonDeformed/model${k}/${folder_name}/result.ply" --input2="TestData/FullNonDeformed/model${k}/mesh_transformed_1.ply" --save_final_path="TestData/FullNonDeformed/model${k}/${folder_name}/final.ply" --save_destination_path="TestData/FullNonDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}

                echo 'Partial Deformed' >> ${file_name} 
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s="PartialDeformed/model${k}/020_0.ply" --t="PartialDeformed/model${k}/104_1.ply" --source_trans="PartialDeformed/model${k}/020_0_se4.h5" --target_trans="PartialDeformed/model${k}/104_1_se4.h5" --matches="PartialDeformed/model${k}/020_104_0_1.npz" --mesh_path="PartialDeformed/model${k}/mesh_020_0.ply" --output="PartialDeformed/model${k}/${folder_name}/result.ply" --output_trans="PartialDeformed/model${k}/${folder_name}/result_se4.h5" --intermediate_output_folder="PartialDeformed/model${k}/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_center} --average_distance_multiplier=${average_distance_multiplier} --number_iterations_custom_filtering=1 --inlier_outlier_thr=${inlier_outlier_thr} --sampling=${sampling} --print_keypoints --show_lepard_inliers >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" --pred="TestData/PartialDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="TestData/PartialDeformed/model${k}/${folder_name}/result.ply" --initial_1="TestData/PartialDeformed/model${k}/020_0.ply" --initial_2="TestData/PartialDeformed/model${k}/104_1.ply" --matches="TestData/PartialDeformed/model${k}/020_104_0_1.npz" --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" --save_final_path="TestData/PartialDeformed/model${k}/${folder_name}/final.ply" --save_destination_path="TestData/PartialDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}

                echo 'Full Deformed' >> ${file_name} 
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s="FullDeformed/model${k}/020.ply" --t="FullDeformed/model${k}/104.ply" --source_trans="FullDeformed/model${k}/020_se4.h5" --target_trans="FullDeformed/model${k}/104_se4.h5" --matches="FullDeformed/model${k}/020_104.npz" --mesh_path="FullDeformed/model${k}/mesh_020.ply" --output="FullDeformed/model${k}/${folder_name}/result.ply" --output_trans="FullDeformed/model${k}/${folder_name}/result_se4.h5" --intermediate_output_folder="FullDeformed/model${k}/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_center} --average_distance_multiplier=${average_distance_multiplier} --number_iterations_custom_filtering=1 --inlier_outlier_thr=${inlier_outlier_thr} --sampling=${sampling} --print_keypoints --show_lepard_inliers  >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/FullDeformed/model${k}/020_se4.h5" --part2="TestData/FullDeformed/model${k}/104_se4.h5" --pred="TestData/FullDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullDeformed/model${k}/${folder_name}/result.ply" --input2="TestData/FullDeformed/model${k}/104.ply" --matches="TestData/FullDeformed/model${k}/020_104.npz" --save_final_path="TestData/FullDeformed/model${k}/${folder_name}/final.ply" --save_destination_path="TestData/FullDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}

                echo 'Partial Non Deformed' >> ${file_name} 
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s="PartialNonDeformed/model${k}/mesh_transformed_0.ply" --t="PartialNonDeformed/model${k}/mesh_transformed_1.ply" --source_trans="PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" --target_trans="PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" --matches="PartialNonDeformed/model${k}/0_1.npz" --mesh_path="PartialNonDeformed/model${k}/mesh_0.ply" --output="PartialNonDeformed/model${k}/${folder_name}/result.ply" --output_trans="PartialNonDeformed/model${k}/${folder_name}/result_se4.h5" --intermediate_output_folder="PartialNonDeformed/model${k}/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_center} --average_distance_multiplier=${average_distance_multiplier} --number_iterations_custom_filtering=1 --inlier_outlier_thr=${inlier_outlier_thr} --sampling=${sampling} --print_keypoints --show_lepard_inliers >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" --pred="TestData/PartialNonDeformed/model${k}/${folder_name}/result_se4.h5" >> ${file_name}
                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="TestData/PartialNonDeformed/model${k}/${folder_name}/result.ply" --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" --save_final_path="TestData/PartialNonDeformed/model${k}/${folder_name}/final.ply" --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder_name}/destination.ply" >> ${file_name}

            fi
        done
    done
done