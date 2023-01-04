base='/home/aiday.kyzy/dataset/Synthetic/FullDeformedData/TestingData/'

config=LNDP_fcgf.yaml
# config=LNDP.yaml

type=fcgf
# type=kpfcn

preprocessing=none
# preprocessing=mutual

# training_data=full_deformed
training_data=partial_deformed
# training_data=pretrained

# epoch=10
epoch=5
# epoch=1
# epoch=null

number_centers=50
average_distance_multiplier=2.0

if [ "$type" == "fcgf" ] ; then
    inlier_outlier_thr=0.01
else
    inlier_outlier_thr=0.01
fi

if [ "$training_data" == "pretrained" ] ; then
	confidence_threshold=0.1
else
	confidence_threshold=0.000001
fi

filename=Testing/all/test_astrivis_full_deformed_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_custom_adm_${average_distance_multiplier}.txt
folder_name=output_full_deformed_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_custom_adm_${average_distance_multiplier}
rm ${filename}
touch ${filename}

# model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
model_numbers=('002' '042' '085' '126' '167' '207')

if [ $type == "kpfcn" ]; then
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
						
				python3 eval_supervised_astrivis.py \
				--config=config/${config} \
				--s="model${k}/transformed/${file_number1}.ply" \
				--t="model${k}/transformed/${file_number2}.ply" \
				--source_trans="model${k}/transformed/${file_number1}_se4.h5" \
				--target_trans="model${k}/transformed/${file_number2}_se4.h5" \
				--matches="model${k}/matches/${file_number1}_${file_number2}.npz" \
				--output="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}.ply" \
				--output_trans="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_se4.h5" \
				--intermediate_output_folder="model${k}/${folder_name}/${file_number1}_${file_number2}/" \
				--base=${base} \
				--print_keypoints \
				--number_centers=${number_centers} \
				--average_distance_multiplier=${average_distance_multiplier} \
				--inlier_outlier_thr=${inlier_outlier_thr} \
				--custom_filtering \
				--reject_outliers=false \
				--confidence_threshold=${confidence_threshold} \
				>> ${filename}
				
				if [ "$?" != "1" ]; then

				python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
				--part1="${base}/model${k}/transformed/${file_number1}_se4.h5" \
				--part2="${base}/model${k}/transformed/${file_number2}_se4.h5" \
				--pred="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_se4.h5" >> ${filename}
				
				python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
				--input1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}.ply" \
				--input2="${base}/model${k}/transformed/${file_number2}.ply" \
				--matches="${base}/model${k}/matches/${file_number1}_${file_number2}.npz" >> ${filename}
				fi
			done
		done
	done
fi

if [ $type == "fcgf" ]; then
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
						
				python3 eval_supervised_astrivis.py \
				--config=config/${config} \
				--s="model${k}/transformed/${file_number1}.ply" \
				--t="model${k}/transformed/${file_number2}.ply" \
				--s_feats="model${k}/transformed/${file_number1}_fcgf.npz" \
				--t_feats="model${k}/transformed/${file_number2}_fcgf.npz" \
				--source_trans="model${k}/transformed/${file_number1}_se4.h5" \
				--target_trans="model${k}/transformed/${file_number2}_se4.h5" \
				--matches="model${k}/matches/${file_number1}_${file_number2}.npz" \
				--output="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}.ply" \
				--output_trans="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_se4.h5" \
				--intermediate_output_folder="model${k}/${folder_name}/${file_number1}_${file_number2}/" \
				--base=${base} \
				--print_keypoints \
				--number_centers=${number_centers} \
				--average_distance_multiplier=${average_distance_multiplier} \
				--inlier_outlier_thr=${inlier_outlier_thr} \
				--custom_filtering \
				--reject_outliers=false \
				--confidence_threshold=${confidence_threshold} \
				>> ${filename}
				
				if [ "$?" != "1" ]; then

				python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
				--part1="${base}/model${k}/transformed/${file_number1}_se4.h5" \
				--part2="${base}/model${k}/transformed/${file_number2}_se4.h5" \
				--pred="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_se4.h5" >> ${filename}
				
				python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
				--input1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}.ply" \
				--input2="${base}/model${k}/transformed/${file_number2}.ply" \
				--matches="${base}/model${k}/matches/${file_number1}_${file_number2}.npz" >> ${filename}
				fi
			done
		done
	done
fi