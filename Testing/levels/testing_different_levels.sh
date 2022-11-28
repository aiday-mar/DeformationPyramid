levels_list=(2 4 6 8 10 12)
file="Testing/levels/testing_levels.txt"
rm ${file} 
touch ${file}

for levels in ${levels_list[@]}; do

        echo "Test - levels : ${levels}" >> ${file}
        folder=levels_${levels}
        rm -rf TestData/FullNonDeformed/${folder}
        mkdir TestData/FullNonDeformed/${folder}
        rm -rf TestData/FullDeformed/${folder}
        mkdir TestData/FullDeformed/${folder}
        rm -rf TestData/PartialNonDeformed/${folder}
        mkdir TestData/PartialNonDeformed/${folder}
        rm -rf TestData/PartialDeformed/${folder}
        mkdir TestData/PartialDeformed/${folder}

        echo 'Full Non Deformed' >> ${file}
        python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullNonDeformed/mesh_transformed_0.ply' --t='FullNonDeformed/mesh_transformed_1.ply' --source_trans='FullNonDeformed/mesh_transformed_0_se4.h5' --target_trans='FullNonDeformed/mesh_transformed_1_se4.h5' --matches='FullNonDeformed/0_1.npz' --output="FullNonDeformed/${folder}/result.ply" --output_trans="FullNonDeformed/${folder}/result_se4.h5" --intermediate_output_folder="FullNonDeformed/${folder}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --levels=${levels} >> ${file}
        python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/FullNonDeformed/mesh_transformed_0_se4.h5" --part2="TestData/FullNonDeformed/mesh_transformed_1_se4.h5" --pred="TestData/FullNonDeformed/${folder}/result_se4.h5" >> ${file}
        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullNonDeformed/${folder}/result.ply" --input2='TestData/FullNonDeformed/mesh_transformed_1.ply' >> ${file}

        echo 'Partial Deformed' >> ${file}
        python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialDeformed/020_0.ply' --t='PartialDeformed/104_1.ply' --source_trans='PartialDeformed/020_0_se4.h5' --target_trans='PartialDeformed/104_1_se4.h5' --matches='PartialDeformed/020_104_0_1.npz' --output="PartialDeformed/${folder}/result.ply" --output_trans="PartialDeformed/${folder}/result_se4.h5" --intermediate_output_folder="PartialDeformed/${folder}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --levels=${levels} >> ${file}
        python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/PartialDeformed/020_0_se4.h5" --part2="TestData/PartialDeformed/104_1_se4.h5" --pred="TestData/PartialDeformed/${folder}/result_se4.h5" >> ${file}
        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="TestData/PartialDeformed/${folder}/result.ply" --initial_1='TestData/PartialDeformed/020_0.ply' --initial_2='TestData/PartialDeformed/104_1.ply' --matches='TestData/PartialDeformed/020_104_0_1.npz' --part1="TestData/PartialDeformed/020_0_se4.h5" --part2="TestData/PartialDeformed/104_1_se4.h5" --save_final_path="TestData/PartialDeformed/${folder}/final.ply" --save_destination_path="TestData/PartialDeformed/${folder}/destination.ply" >> ${file}

        echo 'Full Deformed' >> ${file}
        python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullDeformed/020.ply' --t='FullDeformed/104.ply' --source_trans='FullDeformed/020_se4.h5' --target_trans='FullDeformed/104_se4.h5' --matches='FullDeformed/020_104.npz' --output="FullDeformed/${folder}/result.ply" --output_trans="FullDeformed/${folder}/result_se4.h5" --intermediate_output_folder="FullDeformed/${folder}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --levels=${levels} >> ${file}
        python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1='TestData/FullDeformed/020_se4.h5' --part2='TestData/FullDeformed/104_se4.h5' --pred="TestData/FullDeformed/${folder}/result_se4.h5" >> ${file}
        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullDeformed/${folder}/result.ply" --input2='TestData/FullDeformed/104.ply' --matches='TestData/FullDeformed/020_104.npz' --save_final_path="TestData/FullDeformed/${folder}/final.ply" --save_destination_path="TestData/FullDeformed/${folder}/destination.ply" >> ${file}

        echo 'Partial Non Deformed' >> ${file}
        python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialNonDeformed/mesh_transformed_0.ply' --t='PartialNonDeformed/mesh_transformed_1.ply' --source_trans='PartialNonDeformed/mesh_transformed_0_se4.h5' --target_trans='PartialNonDeformed/mesh_transformed_1_se4.h5' --matches='PartialNonDeformed/0_1.npz' --output="PartialNonDeformed/${folder}/result.ply" --output_trans="PartialNonDeformed/${folder}/result_se4.h5" --intermediate_output_folder="PartialNonDeformed/${folder}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --levels=${levels} >> ${file}
        python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1='TestData/PartialNonDeformed/mesh_transformed_0_se4.h5' --part2='TestData/PartialNonDeformed/mesh_transformed_1_se4.h5' --pred="TestData/PartialNonDeformed/${folder}/result_se4.h5" >> ${file}
        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --final="TestData/PartialNonDeformed/${folder}/result.ply" --initial='TestData/PartialNonDeformed/mesh_transformed_0.ply' --part1='TestData/PartialNonDeformed/mesh_transformed_0_se4.h5' --part2='TestData/PartialNonDeformed/mesh_transformed_1_se4.h5' --save_final_path="TestData/PartialNonDeformed/${folder}/final.ply" --save_destination_path="TestData/PartialNonDeformed/${folder}/destination.ply" >> ${file}
done