base='home/aiday.kyzy/dataset/FullSyntheticTestingDataDeformed'

for k in {211..225}
do

final_folder="$base/model$k"
mkdir $base/model$k/output
arr=('020', '041', '062', '104', '125', '146', '188', '209', '230')
length_array=${#arr[@]}
end=$(($length_array - 1))

for i in $(seq 0 $end); do
	start=$((i+1))
	for j in $(seq $start $end); do

		file_number1=${arr[$i]}
		file_number2=${arr[$j]}
        python3 eval_supervised_astrivis.py --s=${base}/model${k}/transformed/${file_number1}.ply --t=${base}/model${k}/transformed/${file_number2}.ply --output=${base}/model${k}/output/${file_number1}_to_${file_number2}.ply --output_trans=${base}/model${k}/output/${file_number1}_to_${file_number2}_se4.h5 --matches=${base}/model${k}/matches/${file_number1}_${file_number2}.npz --source_trans=${base}/model${k}/transformed/${file_number1}_se4.h5 --target_trans=${base}/model${k}/transformed/${file_number2}_se4.h5 --config=config/LNDP.yaml
		python3 ../../../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1=${base}/model${k}/transformed/${file_number1}_se4.h5 --part2=${base}/model${k}/transformed/${file_number2}_se4.h5 --pred=${base}/model${k}/output/${file_number1}_to_${file_number2}_se4.h5
        python3 ../../../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1=${base}/model${k}/transformed/${file_number2}.ply --input2=${base}/model${k}/output/${file_number1}_to_${file_number2}.ply
    done
done

done