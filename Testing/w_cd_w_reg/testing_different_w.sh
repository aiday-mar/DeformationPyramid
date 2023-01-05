type=fcgf
# type=kpfcn

# preprocessing=none
preprocessing=mutual

w_reg_list=(0 0.2 0.4 0.6 0.8 1)
w_cd_list=(0 0.2 0.4 0.6 0.8 1)

# model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
model_numbers=('002' '042' '085' '126' '167' '207')

training_data=full_deformed
# training_data=partial_deformed
# training_data=pretrained

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

file="Testing/w_cd_w_reg/testing_w_pre_${preprocessing}_${type}.txt"
rm ${file} 
touch ${file}

if [ $type == "kpfcn" ]; then
    for k in ${model_numbers[@]}
    do
    for w_reg in ${w_reg_list[@]}; do
        for w_cd in ${w_cd_list[@]}; do

            echo "model ${k}" >> ${file}
            echo "w_reg ${w_reg}" >> ${file}
            echo "w_cd ${w_cd}" >> ${file}
            folder=output_${w_reg}_${w_cd}_pre_${preprocessing}_${type}
            rm -rf TestData/FullNonDeformed/model${k}/${folder}
            mkdir TestData/FullNonDeformed/model${k}/${folder}
            touch TestData/FullNonDeformed/model${k}/${folder}/result.txt
            rm -rf TestData/FullDeformed/model${k}/${folder}
            mkdir TestData/FullDeformed/model${k}/${folder}
            touch TestData/FullDeformed/model${k}/${folder}/result.txt
            rm -rf TestData/PartialNonDeformed/model${k}/${folder}
            mkdir TestData/PartialNonDeformed/model${k}/${folder}
            touch TestData/PartialNonDeformed/model${k}/${folder}/result.txt
            rm -rf TestData/PartialDeformed/model${k}/${folder}
            mkdir TestData/PartialDeformed/model${k}/${folder}
            touch TestData/PartialDeformed/model${k}/${folder}/result.txt

            python3 eval_supervised_astrivis.py \
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
            --w_cd=${w_cd} \
            --w_reg=${w_reg} >> ${file}

            if [ "$?" != "1" ]; then
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            --part1="TestData/FullNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
            --part2="TestData/FullNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
            --pred="TestData/FullNonDeformed/model${k}/${folder}/result_se4.h5" >> ${file}

            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --input1="TestData/FullNonDeformed/model${k}/${folder}/result.ply" \
            --input2="TestData/FullNonDeformed/model${k}/mesh_transformed_1.ply" >> ${file}
            fi

            python3 eval_supervised_astrivis.py \
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
            --w_cd=${w_cd} \
            --w_reg=${w_reg} >> ${file}

            if [ "$?" != "1" ]; then
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
            --part1="TestData/FullDeformed/model${k}/020_se4.h5" \
            --part2="TestData/FullDeformed/model${k}/104_se4.h5" \
            --pred="TestData/FullDeformed/model${k}/${folder}/result_se4.h5" >> ${file}

            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --input1="TestData/FullDeformed/model${k}/${folder}/result.ply" \
            --input2="TestData/FullDeformed/model${k}/104.ply" \
            --matches="TestData/FullDeformed/model${k}/020_104.npz" >> ${file}
            fi

            python3 eval_supervised_astrivis.py \
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
            --w_cd=${w_cd} \
            --w_reg=${w_reg} >> ${file}

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
            --save_partial_path="TestData/PartialDeformed/model${k}/020_041_0_1_matched.ply" >> ${file}
            fi

            python3 eval_supervised_astrivis.py \
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
            --w_cd=${w_cd} \
            --w_reg=${w_reg} >> ${file}

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
            --save_partial_path="TestData/PartialNonDeformed/model${k}/0_1_matched.ply" >> ${file}
            fi
        done
    done
    done
fi

if [ $type == "fcgf" ]; then
    for k in ${model_numbers[@]}
    do
    for w_reg in ${w_reg_list[@]}; do
        for w_cd in ${w_cd_list[@]}; do

            echo "model ${k}" >> ${file}
            echo "w_reg ${w_reg}" >> ${file}
            echo "w_cd ${w_cd}" >> ${file}
            folder=output_${w_reg}_${w_cd}_pre_${preprocessing}_${type}
            rm -rf TestData/FullNonDeformed/model${k}/${folder}
            mkdir TestData/FullNonDeformed/model${k}/${folder}
            touch TestData/FullNonDeformed/model${k}/${folder}/result.txt
            rm -rf TestData/FullDeformed/model${k}/${folder}
            mkdir TestData/FullDeformed/model${k}/${folder}
            touch TestData/FullDeformed/model${k}/${folder}/result.txt
            rm -rf TestData/PartialNonDeformed/model${k}/${folder}
            mkdir TestData/PartialNonDeformed/model${k}/${folder}
            touch TestData/PartialNonDeformed/model${k}/${folder}/result.txt
            rm -rf TestData/PartialDeformed/model${k}/${folder}
            mkdir TestData/PartialDeformed/model${k}/${folder}
            touch TestData/PartialDeformed/model${k}/${folder}/result.txt

            python3 eval_supervised_astrivis.py \
            --config=config/${config} \
            --s="FullNonDeformed/model${k}/mesh_transformed_0.ply" \
            --t="FullNonDeformed/model${k}/mesh_transformed_1.ply" \
            --s_feats="FullNonDeformed/model${k}/mesh_transformed_0_fcgf.npz" \
            --t_feats="FullNonDeformed/model${k}/mesh_transformed_1_fcgf.npz"  \
            --source_trans="FullNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
            --target_trans="FullNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
            --matches="FullNonDeformed/model${k}/0_1.npz" \
            --output="FullNonDeformed/model${k}/${folder}/result.ply" \
            --output_trans="FullNonDeformed/model${k}/${folder}/result_se4.h5" \
            --intermediate_output_folder="FullNonDeformed/model${k}/${folder}/" \
            --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' \
            --w_cd=${w_cd} \
            --w_reg=${w_reg} \
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

            python3 eval_supervised_astrivis.py \
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
            --w_cd=${w_cd} \
            --w_reg=${w_reg} \
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
            --matches="TestData/FullDeformed/model${k}/020_104.npz" >> ${file}
            fi

            python3 eval_supervised_astrivis.py \
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
            --w_cd=${w_cd} \
            --w_reg=${w_reg} \
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
            --save_partial_path="TestData/PartialDeformed/model${k}/020_041_0_1_matched.ply" >> ${file}
            fi

            python3 eval_supervised_astrivis.py \
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
            --w_cd=${w_cd} \
            --w_reg=${w_reg} \
            --preprocessing=${preprocessing} \
            --print_keypoints >> ${file}

            if [ "$?" != "1" ]; then
            python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py  \
            --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
            --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
            --pred="TestData/PartialNonDeformed/model${k}/${folder}/result_se4.h5" >> ${file}

            python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
            --final="TestData/PartialNonDeformed/model${k}/${folder}/result.ply" \
            --initial="TestData/PartialNonDeformed/model${k}/mesh_transformed_0.ply" \
            --part1="TestData/PartialNonDeformed/model${k}/mesh_transformed_0_se4.h5" \
            --part2="TestData/PartialNonDeformed/model${k}/mesh_transformed_1_se4.h5" \
            --save_partial_path="TestData/PartialNonDeformed/model${k}/0_1_matched.ply" >> ${file}
            fi
        done
    done
    done
fi