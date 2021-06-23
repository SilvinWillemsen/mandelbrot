"""

This script runs the methods in optimisation_methods.py that generate the
Mandelbrot sets using different optimisation techniques. The script plots and
saves the results of the sets generated by each individual method. As the 
underlying algorithm is the same (just different optimisations) the plots
should be identical.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import mandelbrot.optimisation_methods as OM


# specify the detail of the simulation
detail = 1000;
rVals = np.linspace(-2.0, 1.0, detail)
iVals = np.linspace(-1.5, 1.5, detail)
    

# iniaialise a list with the names of the different methods in 
# optimisation_methods.py

methods = ['OM.naive(detail, rVals, iVals, res)',
           'OM.jit_func(detail, rVals, iVals, res)',
           'OM.njit_par(detail, rVals, iVals, res)',
           'OM.vectorised(detail, rVals, iVals, res)',
           'OM.jit_vectorised(detail, rVals, iVals, res)',
           'OM.gu_jit_vectorised(detail, rVals, iVals, res)']


# plot the results?
plotting = True

# save the results to pdf?
saving = False

# run all methods
for m in methods:
    
    # retrieve name of the function from methods
    function_name = m.replace('OM.', '').replace('(detail, rVals, iVals, res)','')
    print('Computing optimisation_methods.' + function_name + f' with {detail:d} x {detail:d} values for c...')
    
    # reset the result matrix
    res = np.zeros ((detail, detail))
    
    # time and evaluate the method
    tic = time.time()
    res = eval(m)
    toc = time.time() - tic
    print(function_name + f'computed in {toc:3.3} seconds!')

    # plot and save the results
    if plotting:
        fig = plt.figure(figsize=(10,10))
        print('Plotting result...')
        ax = plt.imshow (res, cmap='hot', extent=[-2.0, 1.0, -1.5, 1.5])
        plt.xlabel('$\\mathfrak{R}(c)$')
        plt.ylabel('$\\mathfrak{I}(c)$')
        plt.title('Mandelbrot set generated using '+function_name+'()')
  
        print('Result plotted!')
                
        print('Showing image...')
        plt.show()
        print('Image shown')
        if saving:
            fig.savefig('run_output/' + function_name + '_output.pdf')
