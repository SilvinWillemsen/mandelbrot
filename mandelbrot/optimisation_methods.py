"""

This file contains multiple methods to optimise the calculation of the 
Mandelbrot set in mandelbrot.py. All functions return a detail :math:`\\times`
detail numpy array containing the output of the functions implementing the 
Mandelbrot set.

Naive implementation:
    - naive() implements the computation of the Mandelbrot set without any
    optimisation techniques. 

Numba, jit:
    - jit_func() is identical to naive() but optimised with numba using the 
    @jit decorator
    
    - njit_par() is identical to naive() but optimised with numba using the
    @njit decorator and parallelised using the (parallel=True) flag.
    
    
Due to the iterative nature of the generation of the mandelbrot set, and the 
if-statement it contains (inevitably causing branch divergence to happen), it 
is hard to vectorise the iterations in the Mandelbrot function itself. One 
could, however, vectorise one level up, i.e., have the calculation for 
multiple values of :math:`c` happen ad the same time. 

Numba, vectorize:
    - vectorised() tries to vectorise the calculation of the Mandelbrot set
    by looping through the imaginary values and calculating multiple real 
    values. The inverse was not done, as the numpy arrays used are 
    C-contiguous, meaning that the values of the rows (one imaginary value
    multiple real values) are stored consecutively in memory. The function 
    calls the naive non-optimised implementation for calculating the
    Mandelbrot set for comparison.
    
    - jit_vectorised() is identical to vectorised but uses the numba-optimised
    implementation of the function calculating the Mandelbrot set, as well as
    being numba-optimised itself using the @jit decorator.
    
    - gu_jit_vectorised() attempts to use general ufuncs to implement 
    vectorisation. 

The non-optimised functions (naive() and vectorised()) have an @profile 
decorator so that we can see what lines of code are heaviest. To profile the
functions run "kernprof -l -v mandelbrot/run.py" from the root of the 
repository.


Finally, the jit_save_z() function returns -- on top of the numpy array 
containing the outputs of the Mandelbrot functions -- the values of :math: `z` 
from the last iteration in that function. To optimise the function, we use 
parallelisation  using the the @njit decorator with the parallel flag set to 
True as this optimisation technique was found to speed up the algorithm most
(see benchmark.py). The jit_save_z() method is only used by plot_z_values.py. 


"""

import sys
sys.path.append('../')

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
def naive(detail, rVals, iVals, res):
    
    """
    
    The 'naive' solution for computing the Mandelbrot set using for-loops.

    INPUT::
        
        detail : int
            How detailed should the simulation be.
            
        rVals : Numpy array of size (detail,)
            The values for the real component of c to iterate over.
            
        iVals : Numpy array of size (detail,)
            The values for the imaginary component of c to iterate over.
            
        res : Numpy array of floats of size (detail, detail)
            Matrix of zeros that will be filled with outputs of the function 
            generating the Mandelbrot set.

    OUTPUT::
        
        res : Numpy array of floats of size (detail, detail)
            Matrix containing the result of the function generating the
            Mandelbrot set for all values of c that this function has iterated 
            over

    """
    
    for i in range(detail):
        for r in range(detail):
            res[i, r] = mb.M(rVals[r] + iVals[i]*1j)
        
    return res
            
@jit
def jit_func(detail, rVals, iVals, res):
   
    """
    
    The same 'naive' solution as the naive() but optimised with numba using 
    the @jit decorator
    
    """
   
    for i in range(detail):
        for r in range(detail):
            res[i, r] = mb.M_jit(rVals[r] + iVals[i]*1j)
        
    return res

@njit(parallel=True)
def njit_par(detail, rVals, iVals, res):
    
    """
    
    The same 'naive' solution as the naive() but optimised with numba using 
    parallelisation with the @njit decorator and the parallel flag set to True.
    
    The 'range' funtions have been replaced by 'prange' to tell the compiler
    that these loops can be parallelised.
    
    """
    
    for i in prange(detail):
        for r in prange(detail):
            res[i, r] = mb.M_jit(rVals[r] + iVals[i]*1j)

    return res 


@vectorize(['float32(float32, float32)', 'float64(float64, float64)'])
def _vectorised_loop(r, i):
   
    """
    
    Internal function to be used by vectorised(). The function is vectorised
    using the @vectorise decorator and takes in two floating-point values and
    returns a floating-point value.

    """
    
    return mb.M(r + i*1j)
   
@profile 
def vectorised(detail, rVals, iVals, res):
    
    """
    
    The same 'naive' solution as the naive() but with the nested for-loop
    (looping over the reals) vectorised.
        
    The vectorisation strategy is to calculate a row at a time, rather than a
    column, as the numpy 'matrix' is stored C-contiguously (by default). This 
    means that row elements are neighbouring in memory and the implementation
    should be faster this way.

    """
    
    for i in range(detail):
        res[i, :] = _vectorised_loop(rVals, iVals[i])

    return res 


@vectorize(['float32(float32, float32)', 'float64(float64, float64)'])
def _jit_vectorised_loop(r, i):
   
    """
    
    Internal function to be used by jit_vectorised(). Identical to
    _vectorised_loop(), but uses the @jit version of the function calculating
    the Mandelbrot set.

    """
    
    return mb.M_jit(r + i*1j)


@jit
def jit_vectorised(detail, rVals, iVals, res):
    
    """
    
    Same as the vectorised function, but optimised with numba using the @jit 
    decorator. Furthermore it calls the vectorised loop that uses the numba-
    optimised version of the function calculating the Mandelbrot set.

    """
    
    for i in range(detail):
        res[i, :] = _jit_vectorised_loop(rVals, iVals[i])

    return res 


@guvectorize(['void(int64, float64[:], float64[:], float64[:, :])'], '(), (n),(n)->(n,n)', target='cpu')
def gu_jit_vectorised(detail, rVals, iVals, res):
    
    """
    
    A general ufunc attempting to vectorise both for loops in the jit_func() 
    function using the @guvectorise decorator. Rather than returning the 
    result, it is saved in the last input argument of the function: res. This 
    denoted by the mapping in the argument of the decorator:
        
        (), (n),(n)->(n,n)
        
    which essentially says that it creates a n x n array from a scalar and two
    n x 1 arrays.
    
    """
    
    for i in range(detail):
        for r in range(detail):
            res[i, r] = mb.M_jit(rVals[r] + iVals[i]*1j)



@njit(parallel=True)
def jit_save_z(detail, rVals, iVals, res, z_res, I, T):
    
    """
    
    Extra function that saves the values of :math:`z` from the last iteration
    in the function generating the Mandelbrot set.
    This function is only used by plot_z_values.py. 

    INPUT::
        
        detail : int
            How detailed should the simulation be.
            
        rVals : Numpy array of size (detail,)
            The values for the real component of c to iterate over.
            
        iVals : Numpy array of size (detail,)
            The values for the imaginary component of c to iterate over.
            
        res : Numpy array of floats of size (detail, detail)
            Matrix of zeros that will be filled with outputs of the 
            function generating the Mandelbrot set.

     OUTPUT::
       
        
        z_res : Numpy array of complex128 of size (detail, detail)
            Matrix containing the last value of z before the function
            generating the Mandelbrot set returns.
            
        res : Numpy array of floats of size (detail, detail)
            Matrix containing the result of the function generating the
            Mandelbrot set for all values of c that this function has iterated 
            over.

    """
    for i in prange(detail):
        for r in prange(detail):
            z_res[i, r], res[i, r] = mb.M_save_z(rVals[r] + iVals[i]*1j, I, T)
        
    return z_res, res