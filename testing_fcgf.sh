# type=w_reg
# type=w_cd
type=default

# number=0.5
number=0

# file=eval_supervised_astrivis
file=eval_supervised_astrivis_fcgf

# config=LNDP
config=LDNP_fcgf

mkdir TestData/FullNonDeformed/output_fcgf_${type}_${number}
mkdir TestData/FullDeformed/output_fcgf_${type}_${number}
mkdir TestData/PartialNonDeformed/output_fcgf_${type}_${number}
mkdir TestData/PartialDeformed/output_fcgf_${type}_${number}

python3 ${file}.py --config=config/${config}.yaml --s='FullNonDeformed/mesh_transformed_0.ply' --t='FullNonDeformed/mesh_transformed_1.ply' --source_trans='FullNonDeformed/mesh_transformed_0_se4.h5' --target_trans='FullNonDeformed/mesh_transformed_1_se4.h5' --matches='FullNonDeformed/0_1.npz' --output="FullNonDeformed/output_fcgf_${type}_${number}/result.ply" --output_trans="FullNonDeformed/output_fcgf_${type}_${number}/result_se4.h5" --intermediate_output_folder="FullNonDeformed/output_fcgf_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' > TestData/FullNonDeformed/output_fcgf_${type}_${number}/result.txt
python3 ${file}.py --config=config/${config}.yaml --s='PartialDeformed/020_0.ply' --t='PartialDeformed/104_1.ply' --source_trans='PartialDeformed/020_0_se4.h5' --target_trans='PartialDeformed/104_1_se4.h5' --matches='PartialDeformed/020_104_0_1.npz' --output="PartialDeformed/output_fcgf_${type}_${number}/result.ply" --output_trans="PartialDeformed/output_fcgf_${type}_${number}/result_se4.h5" --intermediate_output_folder="PartialDeformed/output_fcgf_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' > TestData/PartialDeformed/output_fcgf_${type}_${number}/result.txt
python3 ${file}.py --config=config/${config}.yaml --s='FullDeformed/020.ply' --t='FullDeformed/104.ply' --source_trans='FullDeformed/020_se4.h5' --target_trans='FullDeformed/104_se4.h5' --matches='FullDeformed/020_104.npz' --output="FullDeformed/output_fcgf_${type}_${number}/result.ply" --output_trans="FullDeformed/output_fcgf_${type}_${number}/result_se4.h5" --intermediate_output_folder="FullDeformed/output_fcgf_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' > TestData/FullDeformed//output_fcgf_${type}_${number}/result.txt
python3 ${file}.py --config=config/${config}.yaml --s='PartialNonDeformed/mesh_transformed_0.ply' --t='PartialNonDeformed/mesh_transformed_1.ply' --source_trans='PartialNonDeformed/mesh_transformed_0_se4.h5' --target_trans='PartialNonDeformed/mesh_transformed_1_se4.h5' --matches='PartialNonDeformed/0_1.npz' --output="PartialNonDeformed/output_fcgf_${type}_${number}/result.ply" --output_trans="PartialNonDeformed/output_fcgf_${type}_${number}/result_se4.h5" --intermediate_output_folder="PartialNonDeformed/output_fcgf_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' > TestData/PartialNonDeformed/output_fcgf_${type}_${number}/result.txt