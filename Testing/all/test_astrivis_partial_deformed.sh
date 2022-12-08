# config=LNDP_fcgf.yaml
config=LNDP.yaml

#type=fcgf
type=kpfcn

filename=Testing/all/test_astrivis_partial_deformed_${type}.txt
rm ${filename}
touch ${filename}

base='/home/aiday.kyzy/dataset/Synthetic/PartialDeformedData/TestingData'

# model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
model_numbers=('002' '042' '085' '126' '167' '207')

for k in ${model_numbers[@]}
do

# arr=('020' '041' '062' '104' '125' '146' '188' '209' '230')
arr=('020' '062' '125' '188')

mkdir $base/model$k/output
length_array=${#arr[@]}
end=$(($length_array - 1))

for i in $(seq 0 $end); do
	start=$((i+1))
	for j in $(seq $start $end); do

		file_number1=${arr[$i]}
		file_number2=${arr[$j]}
		mkdir $base/model$k/output/${file_number1}_${file_number2}
		echo "model ${k} i ${i} j ${j}" >> ${filename}
		
		# 0 -> 1
		touch ${base}/model${k}/output/${file_number1}_${file_number2}_0_1_se4.h5
		python3 eval_supervised_astrivis.py \
		--config=config/${config} \
		--s="PartialDeformedData/TestingData/model${k}/transformed/${file_number1}_0.ply" \
		--t="PartialDeformedData/TestingData/model${k}/temp_${file_number2}/model_1/cloud/dense.ply" \
		--source_trans="PartialDeformedData/TestingData/model${k}/transformed/${file_number1}_0_se4.h5" \
		--target_trans="identity.h5" \
		--matches="PartialDeformedData/TestingData/model${k}/matches/${file_number1}_${file_number2}_0_1.npz" \
		--output="PartialDeformedData/TestingData/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1.ply" \
		--output_trans="PartialDeformedData/TestingData/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1_se4.h5" \
		--intermediate_output_folder="PartialDeformedData/TestingData/model${k}/output/${file_number1}_${file_number2}/" \
		--print_keypoints >> ${filename}

		if [ "$?" != "1" ]; then
		python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
		--part1="${base}/model${k}/transformed/${file_number1}_0_se4.h5" \
		--part2="identity.h5" \
		--pred="${base}/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1_se4.h5" >> ${filename}

        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
		--input1="${base}/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}_0_1.ply" \
		--input2="PartialDeformedData/TestingData/model${k}/temp_${file_number1}/model_0/cloud/dense.ply" >> ${filename}
		fi
		
		# 1 -> 0
		touch ${base}/model${k}/output/${file_number1}_${file_number2}_1_0_se4.h5
		python3 eval_supervised_astrivis.py \
		--config=config/${config} \
		--s="PartialDeformedData/TestingData/model${k}/transformed/${file_number1}_1.ply" \
		--t="PartialDeformedData/TestingData/model${k}/temp_${file_number2}/model_0/cloud/dense.ply" \
		--source_trans="PartialDeformedData/TestingData/model${k}/transformed/${file_number1}_1_se4.h5" \
		--target_trans="identity.h5" --matches="PartialDeformedData/TestingData/model${k}/matches/${file_number1}_${file_number2}_1_0.npz" \
		--output="PartialDeformedData/TestingData/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}_1_0.ply" \
		--output_trans="PartialDeformedData/TestingData/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}_1_0_se4.h5" \
		--intermediate_ouput_folder="PartialDeformedData/TestingData/model${k}/output/${file_number1}_${file_number2}/" \
		--print_keypoints >> ${filename}
		
		if [ "$?" != "1" ]; then
		python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
		--part1="${base}/model${k}/transformed/${file_number1}_1_se4.h5" \
		--part2="identity.h5" \
		--pred="${base}/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}_1_0_se4.h5" >> ${filename}

        python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
		--input1="${base}/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}_1_0.ply" \
		--input2="PartialDeformedData/TestingData/model${k}/temp_${file_number1}/model_1/cloud/dense.ply" >> ${filename}
		fi
	done
done

done
