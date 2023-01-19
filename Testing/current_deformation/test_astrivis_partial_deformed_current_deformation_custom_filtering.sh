# type=fcgf
type=kpfcn

preprocessing=none
# preprocessing=mutual

training_data=full_deformed
# training_data=partial_deformed
# training_data=pretrained

knn_matching=True
# knn_matching=False

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

if [ "$training_data" == "pretrained" ] ; then
	confidence_threshold=0.01
	confidence_threshold_name=1e-02
else
	confidence_threshold=0.001
	confidence_threshold_name=1e-03
fi

number_centers=50
average_distance_multiplier=2.0
inlier_outlier_thr=0.01

model_numbers=('002' '042' '085' '126' '167' '207')
# model_numbers=('126')

# one_model=True
one_model=False

if [ $knn_matching == "False" ]; then
	coarse_level=-2
	index_coarse_feats=1
else
	coarse_level=-2
	index_coarse_feats=1
fi

if [ "$one_model" == "False" ] ; then
	filename=Testing/current_deformation/test_astrivis_partial_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_knn_${knn_matching}_conf_${confidence_threshold_name}_custom_nc_${number_centers}_adm_${average_distance_multiplier}.txt
	folder_name=output_partial_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_knn_${knn_matching}_conf_${confidence_threshold_name}_custom_nc_${number_centers}_adm_${average_distance_multiplier}
fi

if [ "$one_model" == "True" ] ; then
	filename=Testing/current_deformation/test_astrivis_partial_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_knn_${knn_matching}_conf_${confidence_threshold_name}_custom_nc_${number_centers}_adm_${average_distance_multiplier}_one_model_${model_numbers[0]}.txt
	folder_name=output_partial_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_knn_${knn_matching}_conf_${confidence_threshold_name}_custom_nc_${number_centers}_adm_${average_distance_multiplier}
fi

rm ${filename}
touch ${filename}

base='/home/aiday.kyzy/dataset/Synthetic/PartialDeformedData/TestingData/'

if [ $knn_matching == "False" ]; then
	if [ $type == "kpfcn" ]; then
		for k in ${model_numbers[@]}
		do
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
					--base=$base \
					--confidence_threshold=${confidence_threshold} \
					--only_inference \
					--preprocessing=${preprocessing} \
					--number_centers=${number_centers} \
					--average_distance_multiplier=${average_distance_multiplier} \
					--inlier_outlier_thr=${inlier_outlier_thr} \
					--custom_filtering \
					--reject_outliers=false \
					--print_keypoints >> ${filename}

					if [ "$?" != "1" ]; then
					rm "${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply"

					python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
					--file1="${base}/model${k}/transformed/${file_number1}_0.ply" \
					--file2="${base}/model${k}/transformed/${file_number2}_1.ply" \
					--landmarks1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
					--landmarks2="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
					--save_path="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" >> ${filename}

					python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
					--final="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" \
					--initial_1="${base}/model${k}/transformed/${file_number1}_0.ply" \
					--initial_2="${base}/model${k}/transformed/${file_number2}_1.ply" \
					--matches="${base}/model${k}/matches/${file_number1}_${file_number2}_0_1.npz" \
					--part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" \
					--part2="${base}/model${k}/transformed/${file_number2}_1_se4.h5" >> ${filename}
					fi
				done
			done
		done
	fi

	if [ $type == "fcgf" ]; then
		for k in ${model_numbers[@]}
		do
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
					--s_feats="model${k}/transformed/${file_number1}_0_fcgf.npz" \
					--t_feats="model${k}/transformed/${file_number2}_1_fcgf.npz" \
					--source_trans="model${k}/transformed/${file_number1}_0_se4.h5" \
					--target_trans="model${k}/transformed/${file_number2}_1_se4.h5" \
					--matches="model${k}/matches/${file_number1}_${file_number2}_0_1.npz" \
					--output="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1.ply" \
					--output_trans="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1_se4.h5" \
					--intermediate_output_folder="model${k}/${folder_name}/${file_number1}_${file_number2}/" \
					--base=$base \
					--confidence_threshold=${confidence_threshold} \
					--only_inference \
					--preprocessing=${preprocessing} \
					--number_centers=${number_centers} \
					--average_distance_multiplier=${average_distance_multiplier} \
					--inlier_outlier_thr=${inlier_outlier_thr} \
					--custom_filtering \
					--reject_outliers=false \
					--print_keypoints >> ${filename}

					if [ "$?" != "1" ]; then
					rm "${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply"
					
					python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
					--file1="${base}/model${k}/transformed/${file_number1}_0.ply" \
					--file2="${base}/model${k}/transformed/${file_number2}_1.ply" \
					--landmarks1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
					--landmarks2="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
					--save_path="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" >> ${filename}

					python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
					--final="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" \
					--initial_1="${base}/model${k}/transformed/${file_number1}_0.ply" \
					--initial_2="${base}/model${k}/transformed/${file_number2}_1.ply" \
					--matches="${base}/model${k}/matches/${file_number1}_${file_number2}_0_1.npz" \
					--part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" \
					--part2="${base}/model${k}/transformed/${file_number2}_1_se4.h5" >> ${filename}
					fi
				done
			done
		done
	fi
fi

if [ $knn_matching == "True" ]; then
	if [ $type == "kpfcn" ]; then
		for k in ${model_numbers[@]}
		do
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
					--base=$base \
					--confidence_threshold=${confidence_threshold} \
					--only_inference \
					--preprocessing=${preprocessing} \
					--knn_matching \
					--number_centers=${number_centers} \
					--average_distance_multiplier=${average_distance_multiplier} \
					--inlier_outlier_thr=${inlier_outlier_thr} \
					--custom_filtering \
					--reject_outliers=false \
					--print_keypoints >> ${filename}

					if [ "$?" != "1" ]; then
					rm "${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply"

					python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
					--file1="${base}/model${k}/transformed/${file_number1}_0.ply" \
					--file2="${base}/model${k}/transformed/${file_number2}_1.ply" \
					--landmarks1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_ldmk/s_knn_matching_pcd.ply" \
					--landmarks2="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_ldmk/t_knn_matching_pcd.ply" \
					--save_path="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" >> ${filename}

					python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
					--final="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" \
					--initial_1="${base}/model${k}/transformed/${file_number1}_0.ply" \
					--initial_2="${base}/model${k}/transformed/${file_number2}_1.ply" \
					--matches="${base}/model${k}/matches/${file_number1}_${file_number2}_0_1.npz" \
					--part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" \
					--part2="${base}/model${k}/transformed/${file_number2}_1_se4.h5" >> ${filename}
					fi
				done
			done
		done
	fi

	if [ $type == "fcgf" ]; then
		for k in ${model_numbers[@]}
		do
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
					--s_feats="model${k}/transformed/${file_number1}_0_fcgf.npz" \
					--t_feats="model${k}/transformed/${file_number2}_1_fcgf.npz" \
					--source_trans="model${k}/transformed/${file_number1}_0_se4.h5" \
					--target_trans="model${k}/transformed/${file_number2}_1_se4.h5" \
					--matches="model${k}/matches/${file_number1}_${file_number2}_0_1.npz" \
					--output="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1.ply" \
					--output_trans="model${k}/${folder_name}/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1_se4.h5" \
					--intermediate_output_folder="model${k}/${folder_name}/${file_number1}_${file_number2}/" \
					--base=$base \
					--confidence_threshold=${confidence_threshold} \
					--only_inference \
					--preprocessing=${preprocessing} \
					--knn_matching \
					--number_centers=${number_centers} \
					--average_distance_multiplier=${average_distance_multiplier} \
					--inlier_outlier_thr=${inlier_outlier_thr} \
					--custom_filtering \
					--reject_outliers=false \
					--print_keypoints >> ${filename}

					if [ "$?" != "1" ]; then
					rm "${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply"
					
					python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
					--file1="${base}/model${k}/transformed/${file_number1}_0.ply" \
					--file2="${base}/model${k}/transformed/${file_number2}_1.ply" \
					--landmarks1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_ldmk/s_knn_matching_pcd.ply" \
					--landmarks2="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_ldmk/t_knn_matching_pcd.ply" \
					--save_path="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" >> ${filename}

					python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
					--final="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" \
					--initial_1="${base}/model${k}/transformed/${file_number1}_0.ply" \
					--initial_2="${base}/model${k}/transformed/${file_number2}_1.ply" \
					--matches="${base}/model${k}/matches/${file_number1}_${file_number2}_0_1.npz" \
					--part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" \
					--part2="${base}/model${k}/transformed/${file_number2}_1_se4.h5" >> ${filename}
					fi
				done
			done
		done
	fi
fi
