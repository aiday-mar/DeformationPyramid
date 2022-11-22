# type=w_reg
# type=w_cd
type=default

# number=0.5
number=0

mkdir TestData/FullNonDeformed/output_${type}_${number}
mkdir TestData/FullDeformed/output_${type}_${number}
mkdir TestData/PartialNonDeformed/output_${type}_${number}
mkdir TestData/PartialDeformed/output_${type}_${number}

file_name=TestData/custom_no_preprocessing.txt
rm ${file_name}
touch ${file_name}

echo 'Full Non Deformed' >> ${file_name} 
python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullNonDeformed/mesh_transformed_0.ply' --t='FullNonDeformed/mesh_transformed_1.ply' --source_trans='FullNonDeformed/mesh_transformed_0_se4.h5' --target_trans='FullNonDeformed/mesh_transformed_1_se4.h5' --matches='FullNonDeformed/0_1.npz' --output="FullNonDeformed/output_${type}_${number}/result.ply" --output_trans="FullNonDeformed/output_${type}_${number}/result_se4.h5" --intermediate_output_folder="FullNonDeformed/output_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --preprocessing='none' --print_keypoints --custom_filtering >> ${file_name} # --confidence_threshold=0.1 --preprocessing='none' --index_coarse_feats=1 --coarse_level=-2  --custom_filtering # > TestData/FullNonDeformed/output_${type}_${number}/result.txt 
echo 'Partial Deformed' >> ${file_name} 
python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialDeformed/020_0.ply' --t='PartialDeformed/104_1.ply' --source_trans='PartialDeformed/020_0_se4.h5' --target_trans='PartialDeformed/104_1_se4.h5' --matches='PartialDeformed/020_104_0_1.npz' --output="PartialDeformed/output_${type}_${number}/result.ply" --output_trans="PartialDeformed/output_${type}_${number}/result_se4.h5" --intermediate_output_folder="PartialDeformed/output_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --preprocessing='none' --print_keypoints --custom_filtering >> ${file_name} # --confidence_threshold=0.1 --preprocessing='none' --index_coarse_feats=1 --coarse_level=-2  --custom_filtering # > TestData/PartialDeformed/output_${type}_${number}/result.txt 
echo 'Full Deformed' >> ${file_name} 
python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullDeformed/020.ply' --t='FullDeformed/104.ply' --source_trans='FullDeformed/020_se4.h5' --target_trans='FullDeformed/104_se4.h5' --matches='FullDeformed/020_104.npz' --output="FullDeformed/output_${type}_${number}/result.ply" --output_trans="FullDeformed/output_${type}_${number}/result_se4.h5" --intermediate_output_folder="FullDeformed/output_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --preprocessing='none' --print_keypoints --custom_filtering >> ${file_name} # --confidence_threshold=0.1 --preprocessing='none' --index_coarse_feats=1 --coarse_level=-2  --custom_filtering # > TestData/FullDeformed/output_${type}_${number}/result.txt 
echo 'Partial Non Deformed' >> ${file_name} 
python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialNonDeformed/mesh_transformed_0.ply' --t='PartialNonDeformed/mesh_transformed_1.ply' --source_trans='PartialNonDeformed/mesh_transformed_0_se4.h5' --target_trans='PartialNonDeformed/mesh_transformed_1_se4.h5' --matches='PartialNonDeformed/0_1.npz' --output="PartialNonDeformed/output_${type}_${number}/result.ply" --output_trans="PartialNonDeformed/output_${type}_${number}/result_se4.h5" --intermediate_output_folder="PartialNonDeformed/output_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --preprocessing='none' --print_keypoints --custom_filtering >> ${file_name} # --confidence_threshold=0.1 --preprocessing='none' --index_coarse_feats=1 --coarse_level=-2 --custom_filtering # > TestData/PartialNonDeformed/output_${type}_${number}/result.txt 