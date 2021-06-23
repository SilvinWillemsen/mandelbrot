"""
Tests the various functions in optimisation_methods.py using unittest. 

Testing is important here as we are trying to optimise an algorithm, which
which might cause us to make mistakes in the implementation. 

"""

import unittest
import numpy as np
import sys
sys.path.append('../')


import mandelbrot.optimisation_methods as OM
import h5py

class TestMandelbrot(unittest.TestCase):

    def setUp(self):
        # This will contain a list of the methods we will test
        self.methods = ['OM.naive(detail, rVals, iVals, res)',
                        'OM.jit_func(detail, rVals, iVals, res)',
                        'OM.njit_par(detail, rVals, iVals, res)',
                        'OM.vectorised(detail, rVals, iVals, res)',
                        'OM.jit_vectorised(detail, rVals, iVals, res)',
                        'OM.gu_jit_vectorised(detail, rVals, iVals, res)' ]

        
    def test_optimisation_methods(self):
        
        # load data 
        data_file = h5py.File('naive_output_100x100.hdf5', 'r')
        true_data = data_file['mandelbrot'][...]
        data_file.close()
        
        # specify (low) detail 
        detail = 100
        
        # initialise real and imaginary values to iterate over
        rVals = np.linspace(-2.0, 1.0, detail)
        iVals = np.linspace(-1.5, 1.5, detail)
        
        
        # run all methods and check their outputs against the 'true' data
        for m in self.methods:
            # initialise / reset results matrix
            res = np.zeros((detail, detail))

            print('Testing ' + m + '...')
            res = eval(m)
            self.assertTrue(np.allclose(res, true_data))
            print('Done!')

if __name__ == '__main__':
    unittest.main()
