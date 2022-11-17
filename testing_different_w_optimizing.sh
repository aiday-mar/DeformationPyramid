w_reg_list=(0, 0.2, 0.4, 0.6, 0.8, 1)
w_cd_list=(0, 0.2, 0.4, 0.6, 0.8, 1)

for w_reg in ${w_reg_list[@]}; do
    for w_cd in ${w_cd_list[@]}; do
        mkdir TestData/FullNonDeformed/output_${w_reg}_${w_cd}
        mkdir TestData/FullDeformed/output_${w_reg}_${w_cd}
        mkdir TestData/PartialNonDeformed/output_${w_reg}_${w_cd}
        mkdir TestData/PartialDeformed/output_${w_reg}_${w_cd}

        python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullNonDeformed/mesh_transformed_0.ply' --t='FullNonDeformed/mesh_transformed_1.ply' --source_trans='FullNonDeformed/mesh_transformed_0_se4.h5' --target_trans='FullNonDeformed/mesh_transformed_1_se4.h5' --matches='FullNonDeformed/0_1.npz' --output="FullNonDeformed/output_${w_reg}_${w_cd}/result.ply" --output_trans="FullNonDeformed/output_${w_reg}_${w_cd}/result_se4.h5" --intermediate_output_folder="FullNonDeformed/output_${w_reg}_${w_cd}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --w_cd=${w_cd} --w_reg=${w_reg} > TestData/FullNonDeformed/output_${w_reg}_${w_cd}/result.txt 
        python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialDeformed/020_0.ply' --t='PartialDeformed/104_1.ply' --source_trans='PartialDeformed/020_0_se4.h5' --target_trans='PartialDeformed/104_1_se4.h5' --matches='PartialDeformed/020_104_0_1.npz' --output="PartialDeformed/output_${w_reg}_${w_cd}/result.ply" --output_trans="PartialDeformed/output_${w_reg}_${w_cd}/result_se4.h5" --intermediate_output_folder="PartialDeformed/output_${w_reg}_${w_cd}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --w_cd=${w_cd} --w_reg=${w_reg} > TestData/PartialDeformed/output_${w_reg}_${w_cd}/result.txt 
        python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullDeformed/020.ply' --t='FullDeformed/104.ply' --source_trans='FullDeformed/020_se4.h5' --target_trans='FullDeformed/104_se4.h5' --matches='FullDeformed/020_104.npz' --output="FullDeformed/output_${w_reg}_${w_cd}/result.ply" --output_trans="FullDeformed/output_${w_reg}_${w_cd}/result_se4.h5" --intermediate_output_folder="FullDeformed/output_${w_reg}_${w_cd}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --w_cd=${w_cd} --w_reg=${w_reg} > TestData/FullDeformed//output_${w_reg}_${w_cd}/result.txt 
        python3 eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialNonDeformed/mesh_transformed_0.ply' --t='PartialNonDeformed/mesh_transformed_1.ply' --source_trans='PartialNonDeformed/mesh_transformed_0_se4.h5' --target_trans='PartialNonDeformed/mesh_transformed_1_se4.h5' --matches='PartialNonDeformed/0_1.npz' --output="PartialNonDeformed/output_${w_reg}_${w_cd}/result.ply" --output_trans="PartialNonDeformed/output_${w_reg}_${w_cd}/result_se4.h5" --intermediate_output_folder="PartialNonDeformed/output_${w_reg}_${w_cd}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --w_cd=${w_cd} --w_reg=${w_reg} > TestData/PartialNonDeformed/output_${w_reg}_${w_cd}/result.txt
    done
done
