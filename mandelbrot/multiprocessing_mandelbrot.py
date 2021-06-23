#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file uses the multiprocessing module to divide the tasks when computing 
the Mandelbrot set over several processing units.

As opposed to the other optimisation techniques, all values of :math:`c` are
initialised in a numpy array after which its rows are used for multiprocessing.
Similar to what has been discussed in optimisation_methods.py, rows are chosen
rather than columns as the array is C-contiguous.

The explanation of the code can be found in the comments in-code.

"""

import os, sys, time 
   
import numpy as np
from numpy import matlib as ml
import multiprocessing as mp
import matplotlib.pyplot as plt
from mandelbrot.mandelbrot_alg import M
import pandas as pd


if __name__ == '__main__': # Necessary to make multiprocessing work

    # indicate number of processing units you want to use for calculating the mandelbrot set
    number_of_processing_units = 8

    # Raise an exception if the number of processing units defined exceeds the 
    # number of units available
    if number_of_processing_units > mp.cpu_count():
        raise Exception('Number of processing units wanted exceeds number units available')
    
    # define whether to process asynchronously or not 
    calc_async = True;
    
    # If we want to save time it took to run the simulation
    save_to_csv = False
    
    # decide whether to reset the csv file
    reset_csv = False;
    
    # Create worker pool
    print("Parent process id: {:7d}".format(os.getpid()))
    pool = mp.Pool(processes=number_of_processing_units)
    
    # set how detailed the Mandelbrot set should be
    detail = 5000
    
    # initialise the resulting set
    res = np.zeros ((detail, detail))
    
    # create all used values for c
    print('Creating c matrix...')

    # initialise real and imaginary values to use for c    
    rVals = np.linspace(-2.0, 1.0, detail)
    iVals = np.linspace(-1.5, 1.5, detail)
    
    # create matrices containing the real and imaginary parts of the values for c...
    real_part = ml.repmat(rVals, detail, 1)
    imag_part = ml.repmat(iVals.T, detail, 1).T

    # ... and add them together
    c = real_part + imag_part * 1j
    print('c matrix created!')

    # calculate the Mandelbrot set
    if calc_async:
        
        print('Calculate Mandelbrot set asychronously...')
        tic = time.time()
        for i in range(detail):
            res[i, :] = pool.map_async(M, c[i,:]).get()
            
        toc = time.time() - tic
        print(f'Calculated Mandelbrot asynchronously set in {toc:5.3} seconds!')
        
    else:
    
        print('Calculate Mandelbrot set...')
        tic = time.time()
        for i in range(detail):
            res[i, :] = pool.map(M, c[i,:])        
    
        toc = time.time() - tic
        print(f'Calculated Mandelbrot set in {toc:5.3} seconds!')

    # finalise the pool
    pool.close()
    pool.join()
    
    # plot the results
    print('Plotting result...')
    fig = plt.figure(figsize=(10,10))
    plt.imshow (res, cmap='hot')
    plt.show()
    fig.savefig('multiprocessing_output/mandelbrot_multi_proc.pdf')
    print('Result plotted!')


    # only runs when we want to generate data to plot later (this has already
    # been done on my machine, but I wanted to leave this in)
    
    if save_to_csv:
        # if we want to reset the csv file
        if reset_csv:
            # initialise dataframe
            first_entry = [[number_of_processing_units, toc]]
        
            # put variables in a pandas.DataFrame
            dataframe = pd.DataFrame (first_entry, columns=['Processors', 'Time [s]', 'Asynchronously'])
        else:
        # otherwise, append the already existing dataset
            dataframe = pd.read_csv('time_processors.csv', usecols=['Processors', 'Time [s]', 'Asynchronously'])
            data = {'Processors' : number_of_processing_units, 'Time [s]' : toc, 'Asynchronously' : calc_async}
            dataframe = dataframe.append(data, ignore_index=True) 
        
        # save dataframe to .csv
        dataframe.to_csv('time_processors.csv')

    
