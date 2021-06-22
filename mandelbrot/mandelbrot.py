"""

"""

import numpy as np
import matplotlib.pyplot as plt
import mandelbrot
import time


def M(c, I = 100, T = 2):
    
    """
    Function :math:`\\mathcal{M}(c)` implementing the iterative algorithm to 
    determine whether a complex number :math:`c` is part of the Mandelbrot 
    set. The iterative algorithm is:
        
        :math:`z_{i+1} = z_i^2 + c`
    
    with complex number :math:`z` and iteration number :math:`i`.
    
    If, after :math:`I` iterations :math:`|z|` has not crossed threshold 
    :math:`T`, :math:`c` is part of the Mandelbrot set and the function 
    returns
    
        :math:`\\mathcal{M}(c) = 1`.
    
    If instead, :math:`|z|` has crossed the threshold, the function returns 
    
        :math:`\\mathcal{M}(c) = \\frac{i+1}{I}`.
        
    The lower this number is, the earlier :math:`|z|` has exceeded the
    threshold and the more unstable :math:`c` is to the iterative algorithm. 
    These are the n

    INPUT::
        
        c : complex float
            Starting point of iterative algorithm.
            
        I : number of iterations
            DESCRIPTION.
            
        T : Threshold
            DESCRIPTION.

    OUTPUT::
        
        A floating point value between 0 and 1. The closer to 0 the return 
        value is, the earlier |z| has crossed threshold T and thus the more
        unstable c is to the iterative algorithm

    """
    z = 0
    
    for i in range(I):
        z = z*z + c
        if abs(z) > T:
            return(i+1) / I
    
    # if |z| has not exceeded threshold T, return I / I = 1
    return 1

def naive_solution(detail, rVals, iVals, res):
    
    for r in range(detail):
        for i in range(detail):
            res[i, r] = (M(rVals[r] + iVals[i]*1j, 100, 2))
            
    return res
            
detail = 5000;
rVals = np.linspace(-2.0, 1.0, detail)
iVals = np.linspace(-1.5, 1.5, detail)
    
res = np.zeros ((detail, detail))
tic = time.time()
res = naive_solution(detail, rVals, iVals, res)
toc = time.time() - tic
print(f'The naive computation of the mandelbrot set for {detail:d} x {detail:d} \
      values of c took {toc:1.3} seconds')

plt.imshow (res, cmap='hot')
