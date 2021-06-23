"""

Generate "true" mandelbrot set using the naive algorithm to compare 
optimised methods against. We use a low detail here as this will already show
whether the optimisation has affected the results

"""
import sys
sys.path.append('../')

import h5py
import numpy as np
import mandelbrot.optimisation_methods as OM


# specify (low) detail 
detail = 100

# initialise real and imaginary values to iterate over
rVals = np.linspace(-2.0, 1.0, detail)
iVals = np.linspace(-1.5, 1.5, detail)

# initialise results matrix
res = np.zeros((detail, detail))
          
# run the naive implementation     
OM.naive(detail, rVals, iVals, res)

# save to file
f = h5py.File(f'naive_output_{detail:d}x{detail:d}.hdf5', 'w')
f.create_dataset('mandelbrot', data = res)
f.close()