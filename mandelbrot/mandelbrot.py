"""

This file contains several functions that check whether a value :math:`c` is
part of the Mandelbrot set. The following functions are implemented:
    
    - M: non optimised version
    
    - M_jit: numba-optimised version using the @jit decorator
    
    - M_save_z: numba-optimised version using the @jit decorator that also
    returns the last value of :math:`z` before the iteration returns.

The functions are called from the functions in optimisation_methods.py.

The @profile decorator has been added to the non-optimised version of the 
Mandelbrot calculation to see what lines of code are heaviest. It turns out
that (on my machine) the if statement requires the most computation, around 
38.5% of the total and the calculation of :math:`z` takes aaround 33.2% of the 
processing power. From what I see, there is not much we can do to optimise the
code itself further.

For some reason the multiprocessingMandelbrot.py file does not run when the 
profiling functionality is enabled. This is why it is commented out. For 
profiling the code, uncomment this section. 

"""

from numba import jit, njit, prange


# Took the following from Thomas' example to avoid errors when trying to run 
# files 

# Can't run multiprocessingMandelbrot.py if uncommented
# UNCOMMENT THE FOLLOWING FOR PROFILING #### -> <- ####

####

# No-op for use with profiling and test 

# try:
#     @profile
#     def f(x): return x
# except:
#     def profile(func):
#         def inner(*args, **kwargs):
#             return func(*args, **kwargs)
#         return inner

# @profile

####

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
    
    # initialise z
    z = 0
    
    # main loop
    for i in range(I):
        z = z*z + c
        
        # If the magintude exceeds the threshold return the ratio of the
        # current iteration and the total.
        if abs(z) > T:
            return(i+1) / I
    
    # if |z| has not exceeded threshold T, return I / I = 1
    return 1

@jit
def M_jit(c, I = 100, T = 2):
    
    """
    
    Identical function to M(c, I, T) (function above) but with the jit decorator
    
    """
    
    # initialise z
    z = 0
    
    # main loop
    for i in range(I):
        z = z*z + c
        
        # If the magintude exceeds the threshold return the ratio of the
        # current iteration and the total.
        if abs(z) > T:
            return(i+1) / I
    
    # if |z| has not exceeded threshold T, return I / I = 1
    return 1


@jit
def M_save_z(c, I = 100, T = 2):
    
    """
        
    Identical function to M_jit(c, I, T), but also returns the current value 
    for :math:`z`. Used by plot_z_values.py.
    
    """
    
    # initialise z
    z = 0
    
    # main loop
    for i in range(I):
        z = z*z + c
        
        # If the magintude exceeds the threshold return the ratio of the
        # current iteration and the total.
        if abs(z) > T:
            return z, (i+1) / I
    
    # if |z| has not exceeded threshold T, return I / I = 1
    return z, 1