"""

Code to call all our different functions and a basis for possible
profiling

Profiling script. Calls different functions, evaluates and times them.

Every time a function is profiled, it is ran first to 'warm up the computer'
(or something similar that Thomas said during the lecture). 

Why I think it might be necessary (and correct me if I'm wrong) is that memory 
allocation only happens the first time it is run. Then the second iteration, 
the functionly has to overwrite already allocated memory. As we want to time
the functionality of the various functions and not the time it takes to 
allocate memory, we do it this way.
                      
"""

import numpy as np
import time

import mandelbrot as mb

methods = [#'mb.naive_solution(detail, rVals, iVals, res)',
           'mb.jit_naive_solution(detail, rVals, iVals, res)',
           'mb.njit_naive_solution(detail, rVals, iVals, res)']

detail = 1000
rVals = np.linspace(-2.0, 1.0, detail)
iVals = np.linspace(-1.5, 1.5, detail)
    
res = np.zeros ((detail, detail))

# output the detail used for the 
print('detail = {}'.format(detail))

for m in methods:
    # run function once to 'warm up computer'
    y = eval(m)
    
    # time the function
    tic = time.time()
    y = eval(m)
    toc = time.time() - tic

    print(f'{m:30s} : {toc:10.2e} [s]'.format(m, toc))
