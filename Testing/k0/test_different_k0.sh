k0_list = (-11 -10 -9 -8 -7 -6 -5)
rm touch ../../TestData/testing_k0
touch ../../TestData/testing_k0.txt

for k0 in ${k0_list[@]}; do
        rm -rf ../../TestData/FullNonDeformed/output_k0_${k0}
        mkdir ../../TestData/FullNonDeformed/output_k0_${k0}
        rm -rf ../../TestData/FullDeformed/output_k0_${k0}
        mkdir ../../TestData/FullDeformed/output_k0_${k0}
        rm -rf ../../TestData/PartialNonDeformed/output_k0_${k0}
        mkdir ../../TestData/PartialNonDeformed/output_k0_${k0}
        rm -rf ../../TestData/PartialDeformed/output_k0_${k0}
        mkdir ../../TestData/PartialDeformed/output_k0_${k0}

        echo 'Full Non Deformed' >> ../../TestData/testing_k0.txt
        python3 ../../eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullNonDeformed/mesh_transformed_0.ply' --t='FullNonDeformed/mesh_transformed_1.ply' --source_trans='FullNonDeformed/mesh_transformed_0_se4.h5' --target_trans='FullNonDeformed/mesh_transformed_1_se4.h5' --matches='FullNonDeformed/0_1.npz' --output="FullNonDeformed/output_${type}_${number}/result.ply" --output_trans="FullNonDeformed/output_${type}_${number}/result_se4.h5" --intermediate_output_folder="FullNonDeformed/output_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --k0=k0 >> ../../TestData/testing_k0.txt
        echo 'Partial Deformed' >> ../../TestData/testing_k0.txt
        python3 ../../eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialDeformed/020_0.ply' --t='PartialDeformed/104_1.ply' --source_trans='PartialDeformed/020_0_se4.h5' --target_trans='PartialDeformed/104_1_se4.h5' --matches='PartialDeformed/020_104_0_1.npz' --output="PartialDeformed/output_${type}_${number}/result.ply" --output_trans="PartialDeformed/output_${type}_${number}/result_se4.h5" --intermediate_output_folder="PartialDeformed/output_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --k0=k0 >> ../../TestData/testing_k0.txt
        echo 'Full Deformed' >> ../../TestData/testing_k0.txt
        python3 ../../eval_supervised_astrivis.py --config=config/LNDP.yaml --s='FullDeformed/020.ply' --t='FullDeformed/104.ply' --source_trans='FullDeformed/020_se4.h5' --target_trans='FullDeformed/104_se4.h5' --matches='FullDeformed/020_104.npz' --output="FullDeformed/output_${type}_${number}/result.ply" --output_trans="FullDeformed/output_${type}_${number}/result_se4.h5" --intermediate_output_folder="FullDeformed/output_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --k0=k0 >> ../../TestData/testing_k0.txt
        echo 'Partial Non Deformed' >> ../../TestData/testing_k0.txt
        python3 ../../eval_supervised_astrivis.py --config=config/LNDP.yaml --s='PartialNonDeformed/mesh_transformed_0.ply' --t='PartialNonDeformed/mesh_transformed_1.ply' --source_trans='PartialNonDeformed/mesh_transformed_0_se4.h5' --target_trans='PartialNonDeformed/mesh_transformed_1_se4.h5' --matches='PartialNonDeformed/0_1.npz' --output="PartialNonDeformed/output_${type}_${number}/result.ply" --output_trans="PartialNonDeformed/output_${type}_${number}/result_se4.h5" --intermediate_output_folder="PartialNonDeformed/output_${type}_${number}/" --base='/home/aiday.kyzy/code/DeformationPyramid/TestData/' --k0=k0 >> ../../TestData/testing_k0.txt
done