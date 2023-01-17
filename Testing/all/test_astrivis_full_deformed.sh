base='/home/aiday.kyzy/dataset/Synthetic/FullDeformedData/TestingData/'

# type=fcgf
type=kpfcn

# preprocessing=none
preprocessing=mutual

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
	confidence_threshold=0.0001
else
	confidence_threshold=0.0001
fi

model_numbers=('002' '042' '085' '126' '167' '207')

n_deformed_levels=4
n_non_deformed_levels=1
# n_deformed_levels=10
# n_non_deformed_levels=10
w_cd=0
w_reg=0

filename=Testing/all/test_astrivis_full_deformed_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_levels_${n_deformed_levels}_knn_${knn_matching}.txt
folder_name=output_full_deformed_pre_${preprocessing}_${type}_td_${training_data}_e_${epoch}_levels_${n_deformed_levels}_knn_${knn_matching}
rm ${filename}
touch ${filename}

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
					--preprocessing=${preprocessing} \
					--level=${n_deformed_levels} \
					--w_cd=${w_cd} \
					--w_reg=${w_reg} \
					--print_keypoints  >> ${filename}
					
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
					--preprocessing=${preprocessing} \
					--level=${n_deformed_levels} \
					--w_cd=${w_cd} \
					--w_reg=${w_reg} \
					--print_keypoints  >> ${filename}
					
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
					--preprocessing=${preprocessing} \
					--level=${n_deformed_levels} \
					--w_cd=${w_cd} \
					--w_reg=${w_reg} \
					--knn_matching \
					--print_keypoints  >> ${filename}
					
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
					--preprocessing=${preprocessing} \
					--level=${n_deformed_levels} \
					--w_cd=${w_cd} \
					--w_reg=${w_reg} \
					--knn_matching \
					--print_keypoints  >> ${filename}
					
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
fi