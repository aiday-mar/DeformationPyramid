# config=LNDP_fcgf.yaml
config=LNDP.yaml

#type=fcgf
type=kpfcn

filename=Testing/all/test_astrivis_full_non_deformed_${type}.txt
rm ${filename}
touch ${filename}

base='/home/aiday.kyzy/dataset/Synthetic/FullNonDeformedData/TestingData'
model_numbers=('002' '008' '015' '022' '029' '035' '042' '049' '056' '066' '073' '079' '085' '093' '100' '106' '113' '120' '126' '133' '140' '147' '153' '160' '167' '174' '180' '187' '194' '201' '207' '214' '221')
for k in ${model_numbers[@]}
do

    mkdir $base/model$k/output
    touch ${base}/model${k}/output/0_1_se4.h5
    echo "model ${k}" >> ${filname}
    python3 eval_supervised_astrivis.py --config=config/${config} --s="FullNonDeformedData/TestingData/model${k}/mesh_transformed_0.ply" --t="FullNonDeformedData/TestingData/model${k}/mesh_sampled.ply" --source_trans="FullNonDeformedData/TestingData/model${k}/mesh_transformed_0_se4.h5" --target_trans="identity.h5" --matches="FullNonDeformedData/TestingData/model${k}/0_1.npz" --output="FullNonDeformedData/TestingData/model${k}/output/0_1.ply" --output_trans="FullNonDeformedData/TestingData/model${k}/output/0_1_se4.h5" --intermediate_ouput_folder="FullNonDeformedData/TestingData/model${k}/output/" >> ${filname}
    python3 ../../code/sfm/python/graphics/mesh/compute_relative_transformation_error.py --part1="${base}/model${k}/mesh_transformed_0_se4.h5" --part2="identity.h5" --pred="${base}/model${k}/output/0_1_se4.h5" >> ${filname}
    python3 ../../code/sfm/python/graphics/mesh/compute_pointcloud_rmse_ir.py --input1="${base}/model${k}/output/0_1.ply" --input2="${base}/model${k}/mesh_sampled.ply" --matches="${base}/model${k}/0_1.npz" >> ${filname}
done