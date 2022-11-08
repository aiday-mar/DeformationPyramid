base='home/aiday.kyzy/dataset/SyntheticTestingDataDeformed'

for k in {211..225}
do

final_folder=$base'/model'$k
arr=('020', '041', '062', '104', '125', '146', '188', '209', '230')
length_array=${#arr[@]}
end=$(($length_array - 1))

for i in $(seq 0 $end); do
	start=$((i+1))
	for j in $(seq $start $end); do

		file_number1=${arr[$i]}
		file_number2=${arr[$j]}
        python3 shape_transfer_astrivis.py -s=${base}/model${i}/dense1.ply -t=${base}/model${i}/dense2.ply --config=config/LNDP.yaml --directory=astrivis_data/${folder}/model${i}

    done
done

done