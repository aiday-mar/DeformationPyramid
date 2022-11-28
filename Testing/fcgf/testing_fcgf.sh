# type=w_reg
# type=w_cd
type=default

# number=0.5
number=0

# file=../../eval_supervised_astrivis
file=eval_supervised_astrivis_fcgf

# config=LNDP
config=LNDP_fcgf

mkdir TestData/FullNonDeformed/output_fcgf_${type}_${number}
mkdir TestData/FullDeformed/output_fcgf_${type}_${number}
mkdir TestData/PartialNonDeformed/output_fcgf_${type}_${number}
mkdir TestData/PartialDeformed/output_fcgf_${type}_${number}

python3 ${file}.py --config=config/${config}.yaml --s='FullNonDeformed/mesh_transformed_0.ply' --s_feats='FullNonDeformed/mesh_transformed_0_fcgf.npz' --t='FullNonDeformed/mesh_transformed_1.ply' --t_feats='FullNonDeformed/mesh_transformed_1_fcgf.npz' --source_trans='FullNonDeformed/mesh_transformed_0_se4.h5' --target_trans='FullNonDeformed/mesh_transformed_1_se4.h5' --matches='FullNonDeformed/0_1.npz' --output="FullNonDeformed/output_fcgf_${type}_${number}/result.ply" --output_trans="FullNonDeformed/output_fcgf_${type}_${number}/result_se4.h5" --intermediate_output_folder="FullNonDeformed/output_fcgf_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' > TestData/FullNonDeformed/output_fcgf_${type}_${number}/result.txt
python3 ${file}.py --config=config/${config}.yaml --s='PartialDeformed/020_0.ply' --s_feats='PartialDeformed/020_0_fcgf.npz' --t='PartialDeformed/104_1.ply' --t_feats='PartialDeformed/104_1_fcgf.npz' --source_trans='PartialDeformed/020_0_se4.h5' --target_trans='PartialDeformed/104_1_se4.h5' --matches='PartialDeformed/020_104_0_1.npz' --output="PartialDeformed/output_fcgf_${type}_${number}/result.ply" --output_trans="PartialDeformed/output_fcgf_${type}_${number}/result_se4.h5" --intermediate_output_folder="PartialDeformed/output_fcgf_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' > TestData/PartialDeformed/output_fcgf_${type}_${number}/result.txt
python3 ${file}.py --config=config/${config}.yaml --s='FullDeformed/020.ply' --s_feats='FullDeformed/020_fcgf.npz' --t='FullDeformed/104.ply' --t_feats='FullDeformed/104_fcgf.npz' --source_trans='FullDeformed/020_se4.h5' --target_trans='FullDeformed/104_se4.h5' --matches='FullDeformed/020_104.npz' --output="FullDeformed/output_fcgf_${type}_${number}/result.ply" --output_trans="FullDeformed/output_fcgf_${type}_${number}/result_se4.h5" --intermediate_output_folder="FullDeformed/output_fcgf_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' > TestData/FullDeformed//output_fcgf_${type}_${number}/result.txt
python3 ${file}.py --config=config/${config}.yaml --s='PartialNonDeformed/mesh_transformed_0.ply' --s_feats='PartialNonDeformed/mesh_transformed_0_fcgf.npz' --t='PartialNonDeformed/mesh_transformed_1.ply' --t_feats='PartialNonDeformed/mesh_transformed_1_fcgf.npz' --source_trans='PartialNonDeformed/mesh_transformed_0_se4.h5' --target_trans='PartialNonDeformed/mesh_transformed_1_se4.h5' --matches='PartialNonDeformed/0_1.npz' --output="PartialNonDeformed/output_fcgf_${type}_${number}/result.ply" --output_trans="PartialNonDeformed/output_fcgf_${type}_${number}/result_se4.h5" --intermediate_output_folder="PartialNonDeformed/output_fcgf_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' > TestData/PartialNonDeformed/output_fcgf_${type}_${number}/result.txt