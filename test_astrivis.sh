# for i in 018 036 053 057 068 070 082 094 100 101 113 116 141 166 175
for i in 207 210 211 215
# for i in 036 053 057
do
# folder='DynamicStrongMotion'
folder='DynamicFlipMerge'
# folder='DynamicStrongMotionTransformed'
python3 shape_transfer_astrivis.py -s=astrivis_data/${folder}/model${i}/dense1.ply -t=astrivis_data/${folder}/model${i}/dense2.ply --config=config/LNDP.yaml --directory=astrivis_data/${folder}/model${i}
done