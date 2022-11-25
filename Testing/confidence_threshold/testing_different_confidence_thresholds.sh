confidence_threshold_list=(0.04 0.06 0.08 0.1 0.2 0.3 0.4 0.5)
file=testing_confidence_threshold.txt
rm ${file} 
touch ${file}

for confidence_threshold in ${confidence_threshold_list[@]}; do

        folder=confidence_threshold_${confidence_threshold}
        rm -rf ../../TestData/FullNonDeformed/${folder}
        mkdir ../../TestData/FullNonDeformed/${folder}
        rm -rf ../../TestData/FullDeformed/${folder}
        mkdir ../../TestData/FullDeformed/${folder}
        rm -rf ../../TestData/PartialNonDeformed/${folder}
        mkdir ../../TestData/PartialNonDeformed/${folder}
        rm -rf ../../TestData/PartialDeformed/${folder}
        mkdir ../../TestData/PartialDeformed/${folder}

        echo 'Full Non Deformed' >> ${file}
        python3 ../../eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullNonDeformed/mesh_transformed_0.ply' --t='FullNonDeformed/mesh_transformed_1.ply' --source_trans='FullNonDeformed/mesh_transformed_0_se4.h5' --target_trans='FullNonDeformed/mesh_transformed_1_se4.h5' --matches='FullNonDeformed/0_1.npz' --output="FullNonDeformed/${folder}/result.ply" --output_trans="FullNonDeformed/${folder}/result_se4.h5" --intermediate_output_folder="FullNonDeformed/${folder}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence_threshold} >> ${file}
        python3 ../../../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/FullNonDeformed/mesh_transformed_0_se4.h5" --part2="TestData/FullNonDeformed/mesh_transformed_1_se4.h5" --pred="TestData/FullNonDeformed/${folder}/result_se4.h5" >> ${file}
        python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullNonDeformed/${folder}/result.ply" --input2='TestData/FullNonDeformed/mesh_transformed_1.ply' >> ${file}

        echo 'Partial Deformed' >> ${file}
        python3 ../../eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialDeformed/020_0.ply' --t='PartialDeformed/104_1.ply' --source_trans='PartialDeformed/020_0_se4.h5' --target_trans='PartialDeformed/104_1_se4.h5' --matches='PartialDeformed/020_104_0_1.npz' --output="PartialDeformed/${folder}/result.ply" --output_trans="PartialDeformed/${folder}/result_se4.h5" --intermediate_output_folder="PartialDeformed/${folder}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence_threshold} >> ${file}
        python3 ../../../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/PartialDeformed/mesh_transformed_0_se4.h5" --part2="TestData/PartialDeformed/mesh_transformed_1_se4.h5" --pred="TestData/PartialDeformed/${folder}/result_se4.h5" >> ${file}
        python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/PartialDeformed/${folder}/result.ply" --input2='TestData/PartialDeformed/mesh_transformed_1.ply' >> ${file}

        echo 'Full Deformed' >> ${file}
        python3 ../../eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullDeformed/020.ply' --t='FullDeformed/104.ply' --source_trans='FullDeformed/020_se4.h5' --target_trans='FullDeformed/104_se4.h5' --matches='FullDeformed/020_104.npz' --output="FullDeformed/${folder}/result.ply" --output_trans="FullDeformed/${folder}/result_se4.h5" --intermediate_output_folder="FullDeformed/${folder}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence_threshold} >> ${file}
        python3 ../../../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/FullDeformed/mesh_transformed_0_se4.h5" --part2="TestData/FullDeformed/mesh_transformed_1_se4.h5" --pred="TestData/FullDeformed/${folder}/result_se4.h5" >> ${file}
        python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/FullDeformed/${folder}/result.ply" --input2='TestData/FullDeformed/mesh_transformed_1.ply' >> ${file}

        echo 'Partial Non Deformed' >> ${file}
        python3 ../../eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialNonDeformed/mesh_transformed_0.ply' --t='PartialNonDeformed/mesh_transformed_1.ply' --source_trans='PartialNonDeformed/mesh_transformed_0_se4.h5' --target_trans='PartialNonDeformed/mesh_transformed_1_se4.h5' --matches='PartialNonDeformed/0_1.npz' --output="PartialNonDeformed/${folder}/result.ply" --output_trans="PartialNonDeformed/${folder}/result_se4.h5" --intermediate_output_folder="PartialNonDeformed/${folder}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --confidence_threshold=${confidence_threshold} >> ${file}
        python3 ../../../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="TestData/PartialNonDeformed/mesh_transformed_0_se4.h5" --part2="TestData/PartialNonDeformed/mesh_transformed_1_se4.h5" --pred="TestData/PartialNonDeformed/${folder}/result_se4.h5" >> ${file}
        python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="TestData/PartialNonDeformed/${folder}/result.ply" --input2='TestData/PartialNonDeformed/mesh_transformed_1.ply' >> ${file}
done