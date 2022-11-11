import h5py
import numpy as np

f = h5py.File('identity.h5', 'w')
f.create_dataset('transformation', data=np.array(np.identity(4)))
f.close()

h=h5py.File('identity.h5', "r")
print(np.array(h['transformation']))