config=LNDP.yaml
type=kpfcn

filename=Testing/using_gt_ldmks/test_astrivis_partial_deformed_gt_ldmks.txt
folder_name=output_partial_deformed_gt_ldmks
rm ${filename}
touch ${filename}

base='/home/aiday.kyzy/dataset/Synthetic/PartialDeformedData/TestingData/'

# model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
model_numbers=('002' '042' '085' '126' '167' '207')

for k in ${model_numbers[@]}
do
	# arr=('020' '041' '062' '104' '125' '146' '188' '209' '230')
	# arr=('020' '062' '125' '188')
	arr=('020' '104')
	mkdir $base/model$k/${folder_name}
	length_array=${#arr[@]}
	end=$(($length_array - 1))

	for i in $(seq 0 $end); do
		start=$((i+1))
		for j in $(seq $start $end); do

			echo "model ${k} file1 ${arr[$i]} file2 ${arr[$j]}" >> ${filename}
			file_number1=${arr[$i]}
			file_number2=${arr[$j]}
			mkdir $base/model$k/${folder_name}/${file_number1}_${file_number2}
			
			# 0 -> 1
			echo "0 to 1" >> ${filename}
			python3 eval_supervised_astrivis.py \
			--config=config/${config} \
			--s="model${k}/transformed/${file_number1}_0.ply" \
			--t="model${k}/transformed/${file_number2}_1.ply" \
			--source_trans="model${k}/transformed/${file_number1}_0_se4.h5" \
			--target_trans="model${k}/transformed/${file_number2}_1_se4.h5" \
			--matches="model${k}/matches/${file_number1}_${file_number2}_0_1.npz" \
			--output="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1.ply" \
			--output_trans="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1_se4.h5" \
			--intermediate_output_folder="model${k}/${folder_name}/${file_number1}_${file_number2}/" \
			--base=${base} \
			--print_keypoints \
			--use_gt_ldmks \
			>> ${filename}

			if [ "$?" != "1" ]; then
			python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
			--part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" \
			--part2="${base}/model${k}/transformed/${file_number2}_1_se4.h5" \
			--pred="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1_se4.h5" >> ${filename}

			python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
			--final="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1.ply" \
			--initial_1="${base}/model${k}/transformed/${file_number1}_0.ply" \
			--initial_2="${base}/model${k}/transformed/${file_number2}_1.ply" \
			--matches="${base}/model${k}/matches/${file_number1}_${file_number2}_0_1.npz" \
			--part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" \
			--part2="${base}/model${k}/transformed/${file_number2}_1_se4.h5" >> ${filename}
			fi
			
			# 1 -> 0
			echo "1 to 0" >> ${filename}
			python3 eval_supervised_astrivis.py \
			--config=config/${config} \
			--s="model${k}/transformed/${file_number1}_1.ply" \
			--t="model${k}/transformed/${file_number2}_0.ply" \
			--source_trans="model${k}/transformed/${file_number1}_1_se4.h5" \
			--target_trans="model${k}/transformed/${file_number2}_0_se4.h5" \
			--matches="model${k}/matches/${file_number1}_${file_number2}_1_0.npz" \
			--output="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_1_0.ply" \
			--output_trans="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_1_0_se4.h5" \
			--intermediate_output_folder="model${k}/${folder_name}/${file_number1}_${file_number2}/" \
			--base=${base} \
			--print_keypoints \
			--use_gt_ldmks \
			>> ${filename}

			if [ "$?" != "1" ]; then
			python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
			--part1="${base}/model${k}/transformed/${file_number1}_1_se4.h5" \
			--part2="${base}/model${k}/transformed/${file_number2}_0_se4.h5" \
			--pred="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_1_0_se4.h5" >> ${filename}

			python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
			--final="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_1_0.ply" \
			--initial_1="${base}/model${k}/transformed/${file_number1}_1.ply" \
			--initial_2="${base}/model${k}/transformed/${file_number2}_0.ply" \
			--matches="${base}/model${k}/matches/${file_number1}_${file_number2}_1_0.npz" \
			--part1="${base}/model${k}/transformed/${file_number1}_1_se4.h5" \
			--part2="${base}/model${k}/transformed/${file_number2}_0_se4.h5" >> ${filename}
			fi
		done
	done
done