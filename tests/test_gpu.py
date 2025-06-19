import cupy as cp

x = cp.arange(10)
print("CuPy array:", x)
print("GPU sum:", cp.sum(x))