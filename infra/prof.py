# Performance profiling for iterating across a 
# large array in 1000 element steps

import numpy as np
import h5py
import cProfile

# Ordinary numpy array
arr = np.arange(5000000)

# Memory map
nmmarr = np.memmap( shape=arr.shape, filename="./benchmark.nmm", mode='w+')
nmmarr[:] = arr[:]

# hdf5 file
f = h5py.File("./benchmark.hdf5", "w", driver='core')
d = f.create_dataset("mydataset", arr.shape, dtype=arr.dtype)

d[:] = arr[:]
f.close()
f = h5py.File("./benchmark.hdf5", "r", driver='core')

hdarr = f.get('mydataset')


# test function
def run(x, bs=10000):
    for i in range(len(x)//bs):
        sum(x[(i*bs):((i+1)*bs)])

#cProfile.run('run(nmmarr)')  # painfully slow

# cProfile.run('run(arr)')  # clearly the fastest way

# # much faster than the numpy.memmap alternative
# cProfile.run('run(hdarr)')
