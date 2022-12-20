# using pretrained kpfcn based Lepard network

# criterion=none
criterion=angle
# criterion=simple
# criterion=shape
# criterion=disc

# config=LNDP_fcgf.yaml
config=LNDP.yaml

# type=fcgf
type=kpfcn

# preprocessing=none
preprocessing=mutual

file="Testing/exterior_boundary_detection/testing_${criterion}_edge_filtering_pre_${preprocessing}_${type}.txt"
rm ${file}
touch ${file}
# model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
# model_numbers=('002' '022' '042' '066' '085' '106' '126' '147' '167' '187' '207')
model_numbers=('002' '042' '085' '126' '167' '207')
confidence_threshold=0.001

min_dist_thr_fcgf=0.02
min_dist_thr_kpfcn=0.02

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

if [ $type == "kpfcn" ]; then
    for k in ${model_numbers[@]}
    do
        echo "model ${k}" >> ${file}
        folder="${criterion}_edge_filtering_pre_${preprocessing}_${type}"
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
            --source_trans="PartialDeformed/model${k}/020_0_se4.h5" \
            --target_trans="PartialDeformed/model${k}/104_1_se4.h5" \
            --matches="PartialDeformed/model${k}/020_104_0_1.npz" \
            --output="PartialDeformed/model${k}/${folder}/result.ply" \
            --output_trans="PartialDeformed/model${k}/${folder}/result_se4.h5" \
            --intermediate_output_folder="PartialDeformed/model${k}/${folder}/" \
            --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
            --print_keypoints >> ${file}
            
            if [ "$?" != "1" ]; then
            # python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            # --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            # --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \ 
            # --pred="TestData/PartialDeformed/model${k}/${folder}/result_se4.h5" \
            # --base="/home/aiday.kyzy/code/DeformationPyramid/" >> ${file}

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
            --print_keypoints >> ${file}

            if [ "$?" != "1" ]; then
            # python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            # --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
            # --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
            # --pred="TestData/PartialNonDeformed/model${k}/${folder}/result_se4.h5" \ 
            # --base="/home/aiday.kyzy/code/DeformationPyramid/" >> ${file}

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
            --print_keypoints >> ${file}
            
            if [ "$?" != "1" ]; then
            # python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            # --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            # --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \ 
            # --pred="TestData/PartialDeformed/model${k}/${folder}/result_se4.h5" \ 
            # --base="/home/aiday.kyzy/code/DeformationPyramid/" >> ${file}

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
            --print_keypoints >> ${file}

            if [ "$?" != "1" ]; then
            # python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            # --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
            # --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
            # --pred="TestData/PartialNonDeformed/model${k}/${folder}/result_se4.h5" \
            # --base="/home/aiday.kyzy/code/DeformationPyramid/" >> ${file}

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
        folder=edge_filtering_pre_${preprocessing}_${type}
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
            --preprocessing=${preprocessing} \
            --print_keypoints >> ${file}

            if [ "$?" != "1" ]; then
            # python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            # --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            # --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
            # --pred="TestData/PartialDeformed/model${k}/${folder}/result_se4.h5" \ 
            # --base="/home/aiday.kyzy/code/DeformationPyramid/" >> ${file}
            
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
            --preprocessing=${preprocessing} \
            --print_keypoints >> ${file}

            if [ "$?" != "1" ]; then
            # python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            # --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
            # --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
            # --pred="TestData/PartialNonDeformed/model${k}/${folder}/result_se4.h5" \
            # --base="/home/aiday.kyzy/code/DeformationPyramid/" >> ${file}

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
            --min_dist_thr=${min_dist_thr_fcgf} \
            --print_keypoints >> ${file}

            if [ "$?" != "1" ]; then
            # python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            # --part1="TestData/PartialDeformed/model${k}/020_0_se4.h5" \
            # --part2="TestData/PartialDeformed/model${k}/104_1_se4.h5" \
            # --pred="TestData/PartialDeformed/model${k}/${folder}/result_se4.h5" \
            # --base="/home/aiday.kyzy/code/DeformationPyramid/" >> ${file}
            
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
            --print_keypoints >> ${file}

            if [ "$?" != "1" ]; then
            # python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            # --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
            # --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
            # --pred="TestData/PartialNonDeformed/model${k}/${folder}/result_se4.h5" \
            # --base="/home/aiday.kyzy/code/DeformationPyramid/" >> ${file}

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