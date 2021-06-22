#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:37:53 2021

@author: SilvinW
"""

import multiprocessing as mp

import os, time, datetime

def _f(d):
    # Defines the f(d) function (f(d) is a single task)
    tic = datetime.datetime.now()
    time.sleep(float(d)/10.)
    pid = os.getpid()
    print(" _f argument: {:2d}, process id: {:7d} ".format(d, pid))
    print(datetime.datetime.now() - tic)
    return pid

def _callback(dummy):
    # Defines the callback function
    print("Input to callback: {0}".format(dummy))
    print("Callback processid: {0}".format(os.getpid()))
    

if __name__ == '__main__': # We have to use this to make it work in this interactive interpreter
    M = mp.cpu_count()
    print("Parent process id: {:7d}".format(os.getpid()))
    pool = mp.Pool(processes=M)
    
    useAsync = False
    
    if useAsync:
        result = pool.map_async(_f, (30 ,15 ,2), callback=_callback)
    else:
        result = pool.map(_f, (30 ,15 ,2))

    pool.close()
    pool.join()
    print(result)

# import multiprocessing as mp
# import os, time, datetime
# import numpy as np


# def _f(d):
#     # Defines the f(d) function (f(d) is a single task)
#     time.sleep(float(d)/10.)
#     pid = os.getpid()
#     print(" _f argument: {:2d}, process id: {:7d} ".format(d, pid))
#     return pid

# # if __name__ == '__main__': # We have to use this to make it work in this interactive interpreter
# #     M = mp.cpu_count()
# #     print("Parent process id: {:7d}".format(os.getpid()))
# #     pool = mp.Pool(processes=M)
    
# #     useAsync = False
    
# #     if useAsync:
# #         result = pool.map_async(_f, (30 ,15 ,2), callback=_callback)
# #     else:
# #         result = pool.map(_f, (30 ,15 ,2))

# #     pool.close()
# #     pool.join()
# #     print(result)
    
    

# if __name__ == '__main__': # We have to use this to make it work in this interactive interpreter
#     M = mp.cpu_count()    
#     print("Parent process id: {:7d}".format(os.getpid()))
#     pool = mp.Pool(processes=M)
#     result = pool.map_async(_f, (30 ,15 ,2), callback=_callback)
#     pool.close()
#     pool.join()
#     print(result.get())