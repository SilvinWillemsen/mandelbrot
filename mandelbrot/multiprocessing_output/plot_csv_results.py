"""

Script to plot the results saved in the time_processors.csv file. The .csv 
file was generated from the multiprocessingMandelbrot.py file and the plot is 
saved in this folder as multiprocessing_time_results.pdf


Discussion and Conclusion:
    The machine used is a 2017 MacBook Pro with a 2.2 GHz Intel i7 Quad Core 
    processor. 
    
    Even though the machine only has 4 physical cores, it can use 4 additional
    virtual cores for hyperthreading. Looking at the results, it seems that 
    this functionality is either not utilised internally, or the code needs to 
    be written in a different way, as the computational time does not decrease
    significatnly for more than 4 processing units.
    
    Differences between synchronous and asynchronous multiprocessing do not
    seem to be significant. Further investigation needs to be done to confirm 
    that the implementation has been done correctly, or whether asynchronous
    multiprocessing simply does not matter for this specific task.

"""

import pandas as pd
import matplotlib.pyplot as plt


# read the csv
data = pd.read_csv('time_processors.csv', usecols=['Processors', 'Time [s]', 'Asynchronously'])

# retrieve data from multiprocessing synchronously and asynchronously
sync_data = data.loc[data.Asynchronously == 0, 'Time [s]']
async_data = data.loc[data.Asynchronously == 1, 'Time [s]']

# initialise x data
x_data = range(1,9)

# plot figure
fig = plt.figure()
plt.plot(x_data, async_data)
plt.plot(x_data, sync_data)
plt.title('Time to generate $5000 \\times 5000 $ values of the Mandelbrot set')
plt.xlabel('Number of processing units')
plt.ylabel('Time [s]')
plt.legend(['synchronous', 'asynchronous'])
plt.grid()
plt.show()

# save figure
fig.savefig("multiprocessing_time_results.pdf")