config=LNDP_fcgf.yaml
# config=LNDP.yaml

type=fcgf
# type=kpfcn

# preprocessing=none
preprocessing=mutual

training_data=full_deformed
# training_data=partial_deformed
# training_data=pretrained

if [ "$training_data" == "full_deformed" ] ; then
	epoch=10
elif [ "$training_data" == "partial_deformed" ] ; then
	epoch=5
elif [ "$training_data" == "pretrained" ] ; then
	epoch=null
fi

file="Testing/confidence_threshold/testing_confidence_thresholds_pre_${preprocessing}_${type}.txt"
rm ${file}
touch ${file}
# model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
# model_numbers=('002' '022' '042' '066' '085' '106' '126' '147' '167' '187' '207')
model_numbers=('002' '042' '085' '126' '167' '207')

if [ $type == "kpfcn" ]; then
        confidence_threshold_list=(0.04 0.06 0.08 0.1 0.2 0.3 0.4 0.5)

        for k in ${model_numbers[@]}
        do
        for confidence_threshold in ${confidence_threshold_list[@]}; do

                echo "model ${k}" >> ${file}
                echo "Test - confidence threshold : ${confidence_threshold}" >> ${file}
                folder=confidence_threshold_${confidence_threshold}_pre_${preprocessing}_${type}
                rm -rf TestData/FullNonDeformed/model${k}/${folder}
                mkdir TestData/FullNonDeformed/model${k}/${folder}
                rm -rf TestData/FullDeformed/model${k}/${folder}
                mkdir TestData/FullDeformed/model${k}/${folder}
                rm -rf TestData/PartialNonDeformed/model${k}/${folder}
                mkdir TestData/PartialNonDeformed/model${k}/${folder}
                rm -rf TestData/PartialDeformed/model${k}/${folder}
                mkdir TestData/PartialDeformed/model${k}/${folder}


                echo 'Full Non Deformed' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="FullNonDeformed/model${k}/mesh_transformed_0.ply" \
                --t="FullNonDeformed/model${k}/mesh_transformed_1.ply" \
                --source_trans="FullNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --target_trans="FullNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --matches="FullNonDeformed/model${k}/0_1.npz" \
                --output="FullNonDeformed/model${k}/${folder}/result.ply" \
                --output_trans="FullNonDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="FullNonDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --print_keypoints >> ${file}
                
                if [ "$?" != "1" ]; then
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                --part1="TestData/FullNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --part2="TestData/FullNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --pred="TestData/FullNonDeformed/model${k}/${folder}/result_se4.h5" >> ${file}
                
                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                --input1="TestData/FullNonDeformed/model${k}/${folder}/result.ply" \
                --input2="TestData/FullNonDeformed/model${k}/mesh_transformed_1.ply" >> ${file}
                fi

                echo 'Partial Deformed' >> ${file}
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
                --print_keypoints >> ${file}
                
                if [ "$?" != "1" ]; then
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \ 
                --pred="TestData/PartialDeformed/model${k}/${folder}/result_se4.h5" >> ${file}
                
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

                echo 'Full Deformed' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="FullDeformed/model${k}/020.ply" \
                --t="FullDeformed/model${k}/104.ply" \
                --source_trans="FullDeformed/model${k}/020_se4.h5" \
                --target_trans="FullDeformed/model${k}/104_se4.h5" \
                --matches="FullDeformed/model${k}/020_104.npz" \
                --output="FullDeformed/model${k}/${folder}/result.ply" \
                --output_trans="FullDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="FullDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --print_keypoints >> ${file}
                
                if [ "$?" != "1" ]; then
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                --part1="TestData/FullDeformed/model${k}/020_se4.h5" \
                --part2="TestData/FullDeformed/model${k}/104_se4.h5" \
                --pred="TestData/FullDeformed/model${k}/${folder}/result_se4.h5" >> ${file}
                
                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                --input1="TestData/FullDeformed/model${k}/${folder}/result.ply" \
                --input2="TestData/FullDeformed/model${k}/104.ply" \
                --matches="TestData/FullDeformed/model${k}/020_104.npz" \
                --save_final_path="TestData/FullDeformed/model${k}/${folder}/final.ply" \
                --save_destination_path="TestData/FullDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi

                echo 'Partial Non Deformed' >> ${file}
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
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --pred="TestData/PartialNonDeformed/model${k}/${folder}/result_se4.h5" >> ${file}

                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                --final="TestData/PartialNonDeformed/model${k}/${folder}/result.ply" \
                --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --save_final_path="TestData/PartialNonDeformed/model${k}/${folder}/final.ply" \
                --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi
        done
        done
fi

if [ $type == "fcgf" ]; then
        confidence_threshold_list=(5.0e-07 7.5e-07 1.0e-06 2.5e-06)
        for k in ${model_numbers[@]}
        do
        for confidence_threshold in ${confidence_threshold_list[@]}; do

                echo "model ${k}" >> ${file}
                echo "Test - confidence threshold : ${confidence_threshold}" >> ${file}
                folder=confidence_threshold_${confidence_threshold}_pre_${preprocessing}_${type}
                rm -rf TestData/FullNonDeformed/model${k}/${folder}
                mkdir TestData/FullNonDeformed/model${k}/${folder}
                rm -rf TestData/FullDeformed/model${k}/${folder}
                mkdir TestData/FullDeformed/model${k}/${folder}
                rm -rf TestData/PartialNonDeformed/model${k}/${folder}
                mkdir TestData/PartialNonDeformed/model${k}/${folder}
                rm -rf TestData/PartialDeformed/model${k}/${folder}
                mkdir TestData/PartialDeformed/model${k}/${folder}

                echo 'Full Non Deformed' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="FullNonDeformed/model${k}/mesh_transformed_0.ply" \
                --t="FullNonDeformed/model${k}/mesh_transformed_1.ply" \
                --s_feats="FullNonDeformed/model${k}/mesh_transformed_0_fcgf.npz" \
                --t_feats="FullNonDeformed/model${k}/mesh_transformed_1_fcgf.npz" \
                --source_trans="FullNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --target_trans="FullNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --matches="FullNonDeformed/model${k}/0_1.npz" \
                --output="FullNonDeformed/model${k}/${folder}/result.ply" \
                --output_trans="FullNonDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="FullNonDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --preprocessing=${preprocessing} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                --part1="TestData/FullNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --part2="TestData/FullNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --pred="TestData/FullNonDeformed/model${k}/${folder}/result_se4.h5" >> ${file}

                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                --input1="TestData/FullNonDeformed/model${k}/${folder}/result.ply" \
                --input2="TestData/FullNonDeformed/model${k}/mesh_transformed_1.ply" >> ${file}
                fi

                echo 'Partial Deformed' >> ${file}
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
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
                --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
                --pred="TestData/PartialDeformed/model${k}/${folder}/result_se4.h5" >> ${file}
                
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

                echo 'Full Deformed' >> ${file}
                CUDA_LAUNCH_BLOCKING=1 python3 eval_supervised_astrivis.py \
                --config=config/${config} \
                --s="FullDeformed/model${k}/020.ply" \
                --t="FullDeformed/model${k}/104.ply" \
                --s_feats="FullDeformed/model${k}/020_fcgf.npz" \
                --t_feats="FullDeformed/model${k}/104_fcgf.npz" \
                --source_trans="FullDeformed/model${k}/020_se4.h5" \
                --target_trans="FullDeformed/model${k}/104_se4.h5" \
                --matches="FullDeformed/model${k}/020_104.npz" \
                --output="FullDeformed/model${k}/${folder}/result.ply" \
                --output_trans="FullDeformed/model${k}/${folder}/result_se4.h5" \
                --intermediate_output_folder="FullDeformed/model${k}/${folder}/" \
                --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
                --confidence_threshold=${confidence_threshold} \
                --preprocessing=${preprocessing} \
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                --part1="TestData/FullDeformed/model${k}/020_se4.h5" \
                --part2="TestData/FullDeformed/model${k}/104_se4.h5" \
                --pred="TestData/FullDeformed/model${k}/${folder}/result_se4.h5" >> ${file}
                
                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                --input1="TestData/FullDeformed/model${k}/${folder}/result.ply" \
                --input2="TestData/FullDeformed/model${k}/104.ply" \
                --matches="TestData/FullDeformed/model${k}/020_104.npz" \
                --save_final_path="TestData/FullDeformed/model${k}/${folder}/final.ply" \
                --save_destination_path="TestData/FullDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi

                echo 'Partial Non Deformed' >> ${file}
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
                --print_keypoints >> ${file}

                if [ "$?" != "1" ]; then
                python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
                --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --pred="TestData/PartialNonDeformed/model${k}/${folder}/result_se4.h5" >> ${file}

                python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
                --final="TestData/PartialNonDeformed/model${k}/${folder}/result.ply" \
                --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
                --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
                --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
                --save_final_path="TestData/PartialNonDeformed/model${k}/${folder}/final.ply" \
                --save_destination_path="TestData/PartialNonDeformed/model${k}/${folder}/destination.ply" >> ${file}
                fi
        done
        done
fi