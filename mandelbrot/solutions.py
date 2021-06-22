"""



"""

import mandelbrot as mb
from numba import jit, njit, prange, vectorize, guvectorize, float64, int64



# Took the following from Thomas' example to avoid errors when trying to run files
    
# No-op for use with profiling and test 
try:
    @profile
    def f(x): return x
except:
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner

@profile
def naive_solution(detail, rVals, iVals, res):
    """
    
    The 'naive' solution for computing the mandelbrot set using for-loops

    INPUT::
        
        detail : int
            How detailed should the simulation be.
        rVals : Numpy array of size (detail,)
            The values for the real component of :math:`c` to iterate over.
        iVals : Numpy array of size (detail,)
            The values for the imaginary component of :math:`c` to iterate over.
        res : Matrix (Numpy array of Numpy arrays of size (detail, detail))
            Matrix of zeros that will be filled with outputs of the 
            :math:`\\mathcal{M}` function.

    OUTPUT::
        
    res : Matrix (Numpy array of Numpy arrays of size (detail, detail))
            Matrix containing the result of the :math:`\\mathcal{M}` function for all 
            values of :math`c` that this function has iterated over

    """
    
    for r in range(detail):
        for i in range(detail):
            res[i, r] = mb.M(rVals[r] + iVals[i]*1j)
        
    return res
            
@jit
def jit_naive_solution(detail, rVals, iVals, res):
    
    for r in range(detail):
        for i in range(detail):
            res[i, r] = mb.M_jit(rVals[r] + iVals[i]*1j)
        
    return res

@njit(parallel=True)
def njit_par_naive_solution(detail, rVals, iVals, res):
    
    for r in prange(detail):
        for i in prange(detail):
            res[i, r] = mb.M_jit(rVals[r] + iVals[i]*1j)

    return res 

@vectorize(['float32(float32, float32)',
            'float64(float64, float64)'])
def vectorised_loop(r, i):
    return mb.M_jit(r + i*1j)
    
def vectorised_solution(detail, rVals, iVals, res):
    """
    
    The vectorisation strategy is to calculate a row at a time, rather than a
    column, as the numpy 'matrix' is stored C-contiguously (by default). This 
    means that row elements are neighbouring in memory.
    

    """
    
    for i in range(detail):
        res[i, :] = vectorised_loop(rVals, iVals[i])

    return res 

@guvectorize(['void(int64, float64[:], float64[:], float64[:, :])'], '(), (n),(n)->(n,n)', target='cpu')
def gu_vectorised_solution(detail, rVals, iVals, res):
    
    for r in range(detail):
        for i in range(detail):
            res[i, r] = mb.M_jit(rVals[r] + iVals[i]*1j)

