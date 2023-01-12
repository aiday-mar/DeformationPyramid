base='/home/aiday.kyzy/dataset/Synthetic/FullDeformedData/TestingData/'

# type=fcgf
type=kpfcn

# training_data=full_deformed
training_data=partial_deformed
# training_data=pretrained

preprocessing=none
# preprocessing=mutual

# knn_matching=True
knn_matching=False

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
	confidence_threshold=0.1
else
	if [ "$training_data" == "partial_deformed" ]  && ["$type" == "kpfcn" ] ; then
		confidence_threshold=0.0000001
	else
		confidence_threshold=0.000001
	fi
fi

filename=Testing/current_deformation/test_astrivis_full_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_knn_${knn_matching}.txt
folder_name=output_full_deformed_current_deformation_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_knn_${knn_matching}
rm ${filename}
touch ${filename}

model_numbers=('002' '042' '085' '126' '167' '207')

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
					--confidence_threshold=${confidence_threshold} \
					--print_keypoints \
					--preprocessing=${preprocessing} \
					--only_inference >> ${filename}
					
					if [ "$?" != "1" ]; then
					rm "${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply"

					python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
					--file1="${base}/model${k}/transformed/${file_number1}.ply" \
					--file2="${base}/model${k}/transformed/${file_number2}.ply" \
					--landmarks1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
					--landmarks2="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
					--save_path="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" >> ${filename}

					python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
					--input1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" \
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
					--confidence_threshold=${confidence_threshold} \
					--print_keypoints \
					--preprocessing=${preprocessing} \
					--only_inference >> ${filename}
					
					if [ "$?" != "1" ]; then
					rm "${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply"
					
					python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
					--file1="${base}/model${k}/transformed/${file_number1}.ply" \
					--file2="${base}/model${k}/transformed/${file_number2}.ply" \
					--landmarks1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_outlier_ldmk/s_outlier_rejected_pcd.ply" \
					--landmarks2="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_outlier_ldmk/t_outlier_rejected_pcd.ply" \
					--save_path="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" >> ${filename}

					python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
					--input1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" \
					--input2="${base}/model${k}/transformed/${file_number2}.ply" \
					--matches="${base}/model${k}/matches/${file_number1}_${file_number2}.npz" >> ${filename}
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
					--confidence_threshold=${confidence_threshold} \
					--print_keypoints \
					--preprocessing=${preprocessing} \
					--knn_matching \
					--only_inference >> ${filename}
					
					if [ "$?" != "1" ]; then
					rm "${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply"

					python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
					--file1="${base}/model${k}/transformed/${file_number1}.ply" \
					--file2="${base}/model${k}/transformed/${file_number2}.ply" \
					--landmarks1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_ldmk/s_knn_matching_pcd.ply" \
					--landmarks2="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_ldmk/t_knn_matching_pcd.ply" \
					--save_path="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" >> ${filename}

					python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
					--input1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" \
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
					--confidence_threshold=${confidence_threshold} \
					--print_keypoints \
					--preprocessing=${preprocessing} \
					--knn_matching \
					--only_inference >> ${filename}
					
					if [ "$?" != "1" ]; then
					rm "${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply"
					
					python3 ../../code/sfm/python/learning/fusion/fusion_cli.py \
					--file1="${base}/model${k}/transformed/${file_number1}.ply" \
					--file2="${base}/model${k}/transformed/${file_number2}.ply" \
					--landmarks1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_ldmk/s_knn_matching_pcd.ply" \
					--landmarks2="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/${type}_ldmk/t_knn_matching_pcd.ply" \
					--save_path="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" >> ${filename}

					python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
					--input1="${base}/model${k}/${folder_name}/${file_number1}_${file_number2}/current_deformation.ply" \
					--input2="${base}/model${k}/transformed/${file_number2}.ply" \
					--matches="${base}/model${k}/matches/${file_number1}_${file_number2}.npz" >> ${filename}
					fi
				done
			done
		done
	fi
fi