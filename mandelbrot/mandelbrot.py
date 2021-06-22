"""



"""

from numba import jit, njit, prange


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

    INPUT::
        
        c : complex float
            Starting point of iterative algorithm.
            
        I : number of iterations
            Number of iterations.
            
        T : float
            Threshold value.

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

@jit
def M_jit(c, I = 100, T = 2):
    """
    
    Identical function to M(c, I, T) (function above) but with a jit decorator
    
    """
    z = 0
    
    for i in prange(I):
        z = z*z + c
        if abs(z) > T:
            return(i+1) / I
    
    # if |z| has not exceeded threshold T, return I / I = 1
    return 1

@njit(parallel=True)
def M_njit(c, I = 100, T = 2):
    """
    
    Identical function to M(c, I, T) (function above) but with a jit decorator
    
    """
    z = 0
    
    for i in prange(I):
        z = z*z + c
        if abs(z) > T:
            return(i+1) / I
    
    # if |z| has not exceeded threshold T, return I / I = 1
    return 1

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
            res[i, r] = (M(rVals[r] + iVals[i]*1j))
        
    return res
            
@jit
def jit_naive_solution(detail, rVals, iVals, res):
    
    for r in range(detail):
        for i in range(detail):
            res[i, r] = (M_jit(rVals[r] + iVals[i]*1j))
        
    return res

@njit(parallel=True)
def njit_naive_solution(detail, rVals, iVals, res):
    
    for r in prange(detail):
        for i in prange(detail):
            res[i, r] = (M_jit(rVals[r] + iVals[i]*1j))

    return res