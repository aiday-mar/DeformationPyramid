# version=1
version=2

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
number_centers=(100 200 300 400)

# average_distance_multiplier=1
# average_distance_multiplier=2
# average_distance_multiplier=3
# average_distance_multiplier=4
average_distance_multipliers=(1 2 3 4)

for number_center in "${number_centers[@]}"
do
    for average_distance_multiplier in "${average_distance_multipliers[@]}"
    do
        folder_name=output_v_${version}_t_${type}_p_${preprocessing}_c_${confidence}_nc_${number_center}_adm_${average_distance_multiplier}_cl_${coarse_level}_ic_${index_coarse_feats}
        file_name=TestData/v_${version}_t_${type}_p_${preprocessing}_c_${confidence}_nc_${number_center}_adm_${average_distance_multiplier}_cl_${coarse_level}_ic_${index_coarse_feats}.txt

        rm -rf TestData/FullNonDeformed/${folder_name}
        mkdir TestData/FullNonDeformed/${folder_name}
        rm -rf TestData/FullDeformed/${folder_name}
        mkdir TestData/FullDeformed/${folder_name}
        rm -rf TestData/PartialNonDeformed/${folder_name}
        mkdir TestData/PartialNonDeformed/${folder_name}
        rm -rf TestData/PartialDeformed/${folder_name}
        mkdir TestData/PartialDeformed/${folder_name}

        rm ${file_name}
        touch ${file_name}

        if [ "$custom_filtering" = true ] ; then
            echo 'Full Non Deformed' >> ${file_name} 
            python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullNonDeformed/mesh_transformed_0.ply' --t='FullNonDeformed/mesh_transformed_1.ply' --source_trans='FullNonDeformed/mesh_transformed_0_se4.h5' --target_trans='FullNonDeformed/mesh_transformed_1_se4.h5' --matches='FullNonDeformed/0_1.npz' --output="FullNonDeformed/${folder_name}/result.ply" --output_trans="FullNonDeformed/${folder_name}/result_se4.h5" --intermediate_output_folder="FullNonDeformed/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_centers} --average_distance_multiplier=${average_distance_multiplier} --print_keypoints --custom_filtering >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/FullNonDeformed/mesh_transformed_0_se4.h5" --part2="TestData/FullNonDeformed/mesh_transformed_1_se4.h5" --pred="TestData/FullNonDeformed/${folder_name}/result_se4.h5" >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullNonDeformed/${folder_name}/result.ply" --input2='TestData/FullNonDeformed/mesh_transformed_1.ply' --save_final_path="TestData/FullNonDeformed/${folder_name}/final.ply" --save_destination_path="TestData/FullNonDeformed/${folder_name}/destination.ply" >> ${file_name}

            echo 'Partial Deformed' >> ${file_name} 
            python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialDeformed/020_0.ply' --t='PartialDeformed/104_1.ply' --source_trans='PartialDeformed/020_0_se4.h5' --target_trans='PartialDeformed/104_1_se4.h5' --matches='PartialDeformed/020_104_0_1.npz' --output="PartialDeformed/${folder_name}/result.ply" --output_trans="PartialDeformed/${folder_name}/result_se4.h5" --intermediate_output_folder="PartialDeformed/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_centers} --average_distance_multiplier=${average_distance_multiplier} --print_keypoints --custom_filtering >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/PartialDeformed/020_0_se4.h5" --part2="TestData/PartialDeformed/104_1_se4.h5" --pred="TestData/PartialDeformed/${folder_name}/result_se4.h5" >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="TestData/PartialDeformed/${folder_name}/result.ply" --initial_1='TestData/PartialDeformed/020_0.ply' --initial_2='TestData/PartialDeformed/104_1.ply' --matches='TestData/PartialDeformed/020_104_0_1.npz' --part1="TestData/PartialDeformed/020_0_se4.h5" --part2="TestData/PartialDeformed/104_1_se4.h5" --save_final_path="TestData/PartialDeformed/${folder_name}/final.ply" --save_destination_path="TestData/PartialDeformed/${folder_name}/destination.ply" >> ${file_name}

            echo 'Full Deformed' >> ${file_name} 
            python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullDeformed/020.ply' --t='FullDeformed/104.ply' --source_trans='FullDeformed/020_se4.h5' --target_trans='FullDeformed/104_se4.h5' --matches='FullDeformed/020_104.npz' --output="FullDeformed/${folder_name}/result.ply" --output_trans="FullDeformed/${folder_name}/result_se4.h5" --intermediate_output_folder="FullDeformed/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_centers} --average_distance_multiplier=${average_distance_multiplier} --print_keypoints --custom_filtering >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1='TestData/FullDeformed/020_se4.h5' --part2='TestData/FullDeformed/104_se4.h5' --pred="TestData/FullDeformed/${folder_name}/result_se4.h5" >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullDeformed/${folder_name}/result.ply" --input2='TestData/FullDeformed/104.ply' --matches='TestData/FullDeformed/020_104.npz' --save_final_path="TestData/FullDeformed/${folder_name}/final.ply" --save_destination_path="TestData/FullDeformed/${folder_name}/destination.ply" >> ${file_name}

            echo 'Partial Non Deformed' >> ${file_name} 
            python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialNonDeformed/mesh_transformed_0.ply' --t='PartialNonDeformed/mesh_transformed_1.ply' --source_trans='PartialNonDeformed/mesh_transformed_0_se4.h5' --target_trans='PartialNonDeformed/mesh_transformed_1_se4.h5' --matches='PartialNonDeformed/0_1.npz' --output="PartialNonDeformed/${folder_name}/result.ply" --output_trans="PartialNonDeformed/${folder_name}/result_se4.h5" --intermediate_output_folder="PartialNonDeformed/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_centers} --average_distance_multiplier=${average_distance_multiplier} --print_keypoints --custom_filtering >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1='TestData/PartialNonDeformed/mesh_transformed_0_se4.h5' --part2='TestData/PartialNonDeformed/mesh_transformed_1_se4.h5' --pred="TestData/PartialNonDeformed/${folder_name}/result_se4.h5" >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="TestData/PartialNonDeformed/${folder_name}/result.ply" --initial='TestData/PartialNonDeformed/mesh_transformed_0.ply' --part1='TestData/PartialNonDeformed/mesh_transformed_0_se4.h5' --part2='TestData/PartialNonDeformed/mesh_transformed_1_se4.h5' --save_final_path="TestData/PartialNonDeformed/${folder_name}/final.ply" --save_destination_path="TestData/PartialNonDeformed/${folder_name}/destination.ply" >> ${file_name}

        else
            echo 'Full Non Deformed' >> ${file_name} 
            python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullNonDeformed/mesh_transformed_0.ply' --t='FullNonDeformed/mesh_transformed_1.ply' --source_trans='FullNonDeformed/mesh_transformed_0_se4.h5' --target_trans='FullNonDeformed/mesh_transformed_1_se4.h5' --matches='FullNonDeformed/0_1.npz' --output="FullNonDeformed/${folder_name}/result.ply" --output_trans="FullNonDeformed/${folder_name}/result_se4.h5" --intermediate_output_folder="FullNonDeformed/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_centers} --average_distance_multiplier=${average_distance_multiplier} --print_keypoints >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/FullNonDeformed/mesh_transformed_0_se4.h5" --part2="TestData/FullNonDeformed/mesh_transformed_1_se4.h5" --pred="TestData/FullNonDeformed/${folder_name}/result_se4.h5" >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullNonDeformed/${folder_name}/result.ply" --input2='TestData/FullNonDeformed/mesh_transformed_1.ply' --save_final_path="TestData/FullNonDeformed/${folder_name}/final.ply" --save_destination_path="TestData/FullNonDeformed/${folder_name}/destination.ply" >> ${file_name}

            echo 'Partial Deformed' >> ${file_name} 
            python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialDeformed/020_0.ply' --t='PartialDeformed/104_1.ply' --source_trans='PartialDeformed/020_0_se4.h5' --target_trans='PartialDeformed/104_1_se4.h5' --matches='PartialDeformed/020_104_0_1.npz' --output="PartialDeformed/${folder_name}/result.ply" --output_trans="PartialDeformed/${folder_name}/result_se4.h5" --intermediate_output_folder="PartialDeformed/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_centers} --average_distance_multiplier=${average_distance_multiplier} --print_keypoints >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/PartialDeformed/020_0_se4.h5" --part2="TestData/PartialDeformed/104_1_se4.h5" --pred="TestData/PartialDeformed/${folder_name}/result_se4.h5" >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="TestData/PartialDeformed/${folder_name}/result.ply" --initial_1='TestData/PartialDeformed/020_0.ply' --initial_2='TestData/PartialDeformed/104_1.ply' --matches='TestData/PartialDeformed/020_104_0_1.npz' --part1="TestData/PartialDeformed/020_0_se4.h5" --part2="TestData/PartialDeformed/104_1_se4.h5" --save_final_path="TestData/PartialDeformed/${folder_name}/final.ply" --save_destination_path="TestData/PartialDeformed/${folder_name}/destination.ply" >> ${file_name}

            echo 'Full Deformed' >> ${file_name} 
            python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullDeformed/020.ply' --t='FullDeformed/104.ply' --source_trans='FullDeformed/020_se4.h5' --target_trans='FullDeformed/104_se4.h5' --matches='FullDeformed/020_104.npz' --output="FullDeformed/${folder_name}/result.ply" --output_trans="FullDeformed/${folder_name}/result_se4.h5" --intermediate_output_folder="FullDeformed/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_centers} --average_distance_multiplier=${average_distance_multiplier} --print_keypoints >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1='TestData/FullDeformed/020_se4.h5' --part2='TestData/FullDeformed/104_se4.h5' --pred="TestData/FullDeformed/${folder_name}/result_se4.h5" >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullDeformed/${folder_name}/result.ply" --input2='TestData/FullDeformed/104.ply' --matches='TestData/FullDeformed/020_104.npz' --save_final_path="TestData/FullDeformed/${folder_name}/final.ply" --save_destination_path="TestData/FullDeformed/${folder_name}/destination.ply" >> ${file_name}

            echo 'Partial Non Deformed' >> ${file_name} 
            python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialNonDeformed/mesh_transformed_0.ply' --t='PartialNonDeformed/mesh_transformed_1.ply' --source_trans='PartialNonDeformed/mesh_transformed_0_se4.h5' --target_trans='PartialNonDeformed/mesh_transformed_1_se4.h5' --matches='PartialNonDeformed/0_1.npz' --output="PartialNonDeformed/${folder_name}/result.ply" --output_trans="PartialNonDeformed/${folder_name}/result_se4.h5" --intermediate_output_folder="PartialNonDeformed/${folder_name}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence} --preprocessing=${preprocessing} --index_coarse_feats=1 --coarse_level=${coarse_level} --number_centers=${number_centers} --average_distance_multiplier=${average_distance_multiplier} --print_keypoints >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1='TestData/PartialNonDeformed/mesh_transformed_0_se4.h5' --part2='TestData/PartialNonDeformed/mesh_transformed_1_se4.h5' --pred="TestData/PartialNonDeformed/${folder_name}/result_se4.h5" >> ${file_name}
            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="TestData/PartialNonDeformed/${folder_name}/result.ply" --initial='TestData/PartialNonDeformed/mesh_transformed_0.ply' --part1='TestData/PartialNonDeformed/mesh_transformed_0_se4.h5' --part2='TestData/PartialNonDeformed/mesh_transformed_1_se4.h5' --save_final_path="TestData/PartialNonDeformed/${folder_name}/final.ply" --save_destination_path="TestData/PartialNonDeformed/${folder_name}/destination.ply" >> ${file_name}

        fi
    done
done