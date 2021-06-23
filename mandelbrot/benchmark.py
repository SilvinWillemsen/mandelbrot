"""

Profiling script. Calls the different methods in optimisation_methods.py, 
evaluates and times them. Finally, it plots the results. Results for computing
5000 x 5000 values of :math:`c` on my machine (2017 MacBook Pro with a 2.2 Ghz 
Intel i7 processor) can be found in the benchmark_output folder.

Every time a function is profiled, it is ran first to 'warm up the computer'
(or something similar that Thomas said during the lecture). 

Why I think it might be necessary (and correct me if I'm wrong) is that memory 
allocation only happens the first time it is run. Then the second iteration, 
the functionly only has to overwrite already allocated memory. As we want to 
time the functionality of the various methods and not the time it takes to 
allocate memory, we do it this way.

Results and discussion:
    
    Results show that the naive implementation is by far the slowest (as 
    expected).  Vectorisation already helps quite a bit and causes a ca. 3.3x 
    speedup when compared to the non-optimised method. The main speed-up, 
    however, happens by usin!g jit-compilation which speeds the naive 
    implementation up by around x108.3! Vectorisation after using jit 
    compilation (using neither normal nor general ufuncs) does not change 
    the speed by a significant amount. The parallelisation does improve the 
    speed for a total x320.5 speed up from naive to optimised using 
    @njit(parallel=True). 

                      
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import mandelbrot.optimisation_methods as OM

        
# Initialise dataframe to save execution time per method
dataframe = pd.DataFrame ([['', 0]], columns=['Method', 'Time [s]'])
    
# Initialise methods    
methods = ['OM.naive(detail, rVals, iVals, res)',
           'OM.jit_func(detail, rVals, iVals, res)',
           'OM.njit_par(detail, rVals, iVals, res)',
           'OM.vectorised(detail, rVals, iVals, res)',
           'OM.jit_vectorised(detail, rVals, iVals, res)',
           'OM.gu_jit_vectorised(detail, rVals, iVals, res)']

# if we want to save the csv file containing the processing time for each
# individual method and overwrite the current one
save_csv = False

# if we want to save the resulting plots as pdf and overwrite the current ones
save_files = True

# sepecify amount of detail
detail = 5000

# initialise real and imaginary values for c to loop over
rVals = np.linspace(-2.0, 1.0, detail)
iVals = np.linspace(-1.5, 1.5, detail)
    
# initialise results matrix
res = np.zeros ((detail, detail))

# output the detail used for the simulation
print('detail = {}'.format(detail))

for m in methods:
    # run function once to 'warm up computer'
    y = eval(m)
    
    # time the function
    tic = time.time()
    y = eval(m)
    toc = time.time() - tic
    
    # print result
    print(f'{m:30s} : {toc:10.2e} [s]')
    
    # save result to dataframe
    method_name = m.replace('OM.', ''). replace('(detail, rVals, iVals, res)','')
    data = {'Method' : method_name, 'Time [s]' : toc}
    dataframe = dataframe.append(data, ignore_index=True) 

    
# remove the row used for initialising dataframe
dataframe = dataframe.drop(0)

# sort values from slow to fast
dataframe = dataframe.sort_values('Time [s]', 0, False)

# print speed up from naive to method
for m in methods:
    method_name = m.replace('OM.', ''). replace('(detail, rVals, iVals, res)','')
    if method_name == 'naive':
        continue
    speed_up = dataframe.loc[dataframe.Method == 'naive', 'Time [s]'].values / dataframe.loc[dataframe.Method == method_name, 'Time [s]'].values
    print('Speed up from naive to ' + method_name + f' is x{speed_up[0]:3.1f}')

# create x data
x_data = range(1,len(methods)+1)

# plot results linearly
fig = plt.figure(figsize=(7,5))
fig.add_axes([0.1, 0.25, 0.85, 0.7])

plt.plot(x_data, dataframe.loc[:,'Time [s]'])
plt.xticks(x_data, dataframe.loc[:, 'Method'], rotation=45)
plt.xlabel('Method')
plt.ylabel('Time [s]')
plt.grid()
plt.title(f'Time to compute {detail:d} x {detail:d} values of the Mandelbrot set (lin)')
plt.show()

# save figure
if save_files:
    fig.savefig('benchmark_output/time_per_method_linear.pdf')


# plot results logarithmically
fig = plt.figure(figsize=(7,5))
fig.add_axes([0.1, 0.25, 0.85, 0.7])

plt.semilogy(x_data, dataframe.loc[:,'Time [s]'])
plt.xticks(x_data,dataframe.loc[:, 'Method'], rotation=45)
plt.xlabel('Method')
plt.ylabel('Time [s]')
plt.grid()
plt.title(f'Time to compute {detail:d} x {detail:d} values of the Mandelbrot set (log)')
plt.show()

# save figure
if save_files:
    fig.savefig('benchmark_output/time_per_method_logarithmic.pdf')

# overwrite the csv file if specified at the beginning
if save_csv:
    dataframe.to_csv('benchmark_output/methods_benchmark.csv')
