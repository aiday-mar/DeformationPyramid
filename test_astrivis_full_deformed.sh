base='home/aiday.kyzy/dataset/Synthetic/FullDeformedData/TestingData'
# base='home/aiday.kyzy/dataset/Synthetic/FullNonDeformedData/TestingData'
model_numbers=('002', '008', '015', '022', '029', '035', '042', '049', '056', '066', '073', '079', '085', '093', '100', '106', '113', '120', '126', '133', '140', '147', '153', '160', '167', '174', '180', '187', '194', '201', '207', '214', '221')
for k in model_numbers
do

arr=('020', '041', '062', '104', '125', '146', '188', '209', '230')
mkdir $base/model$k/output
length_array=${#arr[@]}
end=$(($length_array - 1))

for i in $(seq 0 $end); do
	start=$((i+1))
	for j in $(seq $start $end); do

		file_number1=${arr[$i]}
		file_number2=${arr[$j]}

        python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s="FullDeformedData/TestingData/model${k}/transformed/${file_number1}.ply" --t="FullDeformedData/TestingData/model${k}/transformed/${file_number2}.ply" --source_trans="FullDeformedData/TestingData/model${k}/transformed/${file_number1}_se4.h5" --target_trans="FullDeformedData/TestingData/model${k}/transformed/${file_number2}_se4.h5" --matches="FullDeformedData/TestingData/model${k}/matches/${file_number1}_${file_number2}.npz" --output="FullDeformedData/TestingData/model${k}/output/${file_number1}_${file_number2}.ply" --output_trans="FullDeformedData/TestingData/model${k}/output/${file_number1}_${file_number2}_se4.h5"
		python3 ../../../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="${base}/model${k}/transformed/${file_number1}_se4.h5" --part2="${base}/model${k}/transformed/${file_number2}_se4.h5" --pred="${base}/model${k}/output/${file_number1}_${file_number2}_se4.h5"
        python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="${base}/model${k}/output/${file_number1}_${file_number2}.ply" --input2="${base}/model${k}/transformed/${file_number2}.ply"

    done
done

done