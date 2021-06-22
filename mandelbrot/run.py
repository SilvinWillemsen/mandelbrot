import numpy as np
import matplotlib.pyplot as plt
import time
import mandelbrot as mb


detail = 1000;
rVals = np.linspace(-2.0, 1.0, detail)
iVals = np.linspace(-1.5, 1.5, detail)
    
res = np.zeros ((detail, detail))

# # naive solution
# print('Computing naive solution...')
# tic = time.time()
# res = mb.naive_solution(detail, rVals, iVals, res)
# toc = time.time() - tic
# print(f'The naive computation of the mandelbrot set for {detail:d} x {detail:d} values of c took {toc:3.3} seconds')

# print('Plotting result...')
# plt.imshow (res, cmap='hot')
# print('Result plotted!')



# # jit solution
# print('Computing jit solution...')
# tic = time.time()
# res = mb.jit_naive_solution(detail, rVals, iVals, res)
# toc = time.time() - tic
# print(f'The jit computation of the mandelbrot set for {detail:d} x {detail:d} values of c took {toc:3.3} seconds')


# njit solution
print('Computing njit solution...')
tic = time.time()
res = mb.njit_naive_solution(detail, rVals, iVals, res)
toc = time.time() - tic
print(f'The njit computation of the mandelbrot set for {detail:d} x {detail:d} values of c took {toc:3.3} seconds')


print('Plotting result...')
fig = plt.figure(figsize=(10,10))
plt.imshow (res, cmap='hot')
print('Result plotted!')

print('Showing image...')
plt.show()
print('Image shown')

print('Saving figure...')
fig.savefig("mandelbrot.pdf")
print('Figure saved!')
