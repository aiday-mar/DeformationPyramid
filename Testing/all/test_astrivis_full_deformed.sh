base='/home/aiday.kyzy/dataset/Synthetic/FullDeformedData/TestingData'
# config=LNDP_fcgf.yaml
config=LNDP.yaml

#type=fcgf
type=kpfcn

# model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
model_numbers=('002' '042' '085' '126' '167' '207')

for k in ${model_numbers[@]}
do

# arr=('020' '041' '062' '104' '125' '146' '188' '209' '230')
arr=('020' '062' '125' '188')

filename=Testing/all/test_astrivis_full_deformed_${type}.txt
rm ${filename}
touch ${filename}

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
		touch ${base}/model${k}/output/${file_number1}_${file_number2}_se4.h5
		
		python3 eval_supervised_astrivis.py \
		--config=config/${config} \
		--s="FullDeformedData/TestingData/model${k}/transformed/${file_number1}.ply" \
		--t="FullDeformedData/TestingData/model${k}/sampled/${file_number2}.ply" \
		--source_trans="FullDeformedData/TestingData/model${k}/transformed/${file_number1}_se4.h5" \
		--target_trans="identity.h5" \
		--matches="FullDeformedData/TestingData/model${k}/matches/${file_number1}_${file_number2}.npz" \
		--output="FullDeformedData/TestingData/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}.ply" \
		--output_trans="FullDeformedData/TestingData/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}_se4.h5" \
		--intermediate_output_folder="FullDeformedData/TestingData/model${k}/output/${file_number1}_${file_number2}/" \
		--print_keypoints  >> ${filename}
		
		if [ "$?" != "1" ]; then
		python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py \
		--part1="${base}/model${k}/transformed/${file_number1}_se4.h5" \
		--part2="identity.h5" \
		--pred="${base}/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}_se4.h5" >> ${filename}
		
		python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py \
		--input1="${base}/model${k}/output/${file_number1}_${file_number2}/${file_number1}_${file_number2}.ply" \
		--input2="${base}/model${k}/sampled/${file_number2}.ply" \
		--matches="${base}/model${k}/matches/${file_number1}_${file_number2}.npz" >> ${filename}
		fi
    done
done

done
