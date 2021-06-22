"""
Tests the mandelbrot functions using unittest. 

Testing is important here as we are trying to optimise an algorithm, which
which might cause us to make mistakes in the implementation. 

"""

import unittest
from scipy.linalg import toeplitz
import numpy as np

import mandelbrot as mb

class TestMandelbrot(unittest.TestCase):

    def setUp(self):
        # This will contain a list of the methods we will test
        self.methods = ['mb.jit_naive_solution(X)', 
                        'mb.jit_naive_solution(X)']

        
    def test_specific(self):
        X = np.array([[1.0, 2.1, 2.0], [1.8, 1.2, 1.9], [2.0, 2.2, 0.8]])
        yh = np.array([1.0, 2.0, 2.0])

        for m in self.methods:
            y = eval(m)
            self.assertTrue(np.allclose(toeplitz(y), toeplitz(yh)))


if __name__ == '__main__':
    unittest.main()
