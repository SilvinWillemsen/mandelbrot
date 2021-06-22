"""

"""

import numpy as np
import matplotlib.pyplot as plt
import time
import solutions as sol


detail = 1000;
rVals = np.linspace(-2.0, 1.0, detail)
iVals = np.linspace(-1.5, 1.5, detail)
    


methods = ['sol.naive_solution(detail, rVals, iVals, res)',
           'sol.jit_naive_solution(detail, rVals, iVals, res)',
           'sol.njit_par_naive_solution(detail, rVals, iVals, res)',
           'sol.vectorised_solution(detail, rVals, iVals, res)',
           'sol.gu_vectorised_solution(detail, rVals, iVals, res)']

plotting = True
for m in methods:
    function_name = m.replace('sol.', '').replace('(detail, rVals, iVals, res)','')
    print('Computing solutions.' + function_name + '...')
    
    res = np.zeros ((detail, detail))
    tic = time.time()
    res = eval(m)
    toc = time.time() - tic
    print(function_name + f'computed in {toc:3.3} seconds!')

    if plotting:
        print('Plotting result...')
        plt.imshow (res, cmap='hot')
        print('Result plotted!')
        
        print('Showing image...')
        plt.show()
        print('Image shown')

