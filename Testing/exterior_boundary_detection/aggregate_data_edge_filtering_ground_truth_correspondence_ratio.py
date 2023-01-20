import matplotlib.pyplot as plt

criteria = ['simple', 'angle', 'shape', 'disc', 'mesh', 'none']
model_numbers=['002', '042', '085', '126', '167', '207']

training_data='full_deformed'
final_matrices = {}


for model_number in model_numbers:

    plt.clf()
    plt.savefig('Testing/exterior_boundary_detection/')