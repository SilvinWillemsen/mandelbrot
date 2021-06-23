"""

This script retrieves and plots the z values before the function generating 
the Mandelbrot set returns. The plot it generates is absolutely stunning so I
wanted to include this in the project :)

This function allows the number of iterations as well as the threshold to be
changed from this script.

"""


import numpy as np
import matplotlib.pyplot as plt
from mandelbrot.optimisation_methods import jit_save_z
import time

# set detail
detail = 10000

# set number of iterations
I = 100

# set threshold
T = 2

# initialise real and imaginary values to iterate over
rVals = np.linspace(-2.0, 1.0, detail)
iVals = np.linspace(-1.5, 1.5, detail)

# initialise empty matrices for the mandelbrot result as well as the values for z
res = np.zeros ((detail, detail))
z_res = np.zeros ((detail, detail)).astype('complex128')


# generate data
print(f'Computing optimisation_methods.jit_save_z with {detail:d} x {detail:d} values for c...')
tic = time.time()
z_res, res = jit_save_z(detail, rVals, iVals, res, z_res, I, T)
toc = time.time() - tic
print(f'jit_save_z computed in {toc:3.3} seconds!')

# plot z values in huge figure (for more detail :) )
fig = plt.figure(figsize=(100, 100))
print('Plotting result...')
plt.imshow (abs(z_res), cmap='hot')
plt.axis('off')
print('Result plotted!')

print('Showing image...')
plt.show()
print('Image shown')
fig.savefig('plot_z_values_output/plot_z.pdf')

# check whether the Mandelbrot set still looks like it should
fig = plt.figure(figsize=(10, 10))
print('Plotting result...')
plt.imshow (res, cmap='hot', extent=[-2.0, 1.0, -1.5, 1.5])
print('Result plotted!')

print('Showing image...')
plt.show()
print('Image shown')
        
