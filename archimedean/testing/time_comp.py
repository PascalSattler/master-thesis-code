import timeit
import numpy as np

# # Your version
# def original(kx, ky, kz):
#     k = np.array([kx, ky, kz])
#     components_zero = np.isclose(k, 0)
#     k_is_zero = all(components_zero)
#     if not k_is_zero:
#         sign = np.sign(k[~components_zero][0])
#         k *= sign
#         kx, ky, kz = k
#     return kx, ky, kz

# # Loop version
# def optimized(kx, ky, kz):
#     k = np.array([kx, ky, kz])
#     for i in range(3):
#         if not np.isclose(k[i], 0):
#             if k[i] < 0:
#                 k = -k
#             break
#     return k

# # Manual version
# def manual(kx, ky, kz):
#     if not np.isclose(kx, 0):
#         if kx < 0:
#             return -kx, -ky, -kz
#     elif not np.isclose(ky, 0):
#         if ky < 0:
#             return -kx, -ky, -kz
#     elif not np.isclose(kz, 0):
#         if kz < 0:
#             return -kx, -ky, -kz
#     return kx, ky, kz

# # Benchmark
# print(timeit.timeit('original(0, -0.5, 1)', globals=globals(), number=100000))
# # ~0.18s

# print(timeit.timeit('optimized(0, -0.5, 1)', globals=globals(), number=100000))
# # ~0.12s (33% faster)

# print(timeit.timeit('manual(0, -0.5, 1)', globals=globals(), number=100000))





# Parameters
# x = 10
# list_size = 1_000_000
# iterations = 1000

# # Test 1: Direct division with Python list
# def divide_python_list():
#     lst = list(range(list_size))
#     return [item / x for item in lst]

# # Test 2: Multiply by 1/x with Python list
# def multiply_python_list():
#     lst = list(range(list_size))
#     inv_x = 1 / x
#     return [item * inv_x for item in lst]

# # Test 3: Direct division with NumPy
# def divide_numpy():
#     arr = np.arange(list_size)
#     return arr / x

# # Test 4: Multiply by 1/x with NumPy
# def multiply_numpy():
#     arr = np.arange(list_size)
#     inv_x = 1 / x
#     return arr * inv_x

# # Run benchmarks
# print("Benchmarking list normalization (lower is faster)\n")
# print(f"List size: {list_size:,} | Iterations: {iterations}\n")

# time_div_py = timeit.timeit(divide_python_list, number=iterations)
# print(f"Python list - Direct division:     {time_div_py:.4f}s")

# time_mul_py = timeit.timeit(multiply_python_list, number=iterations)
# print(f"Python list - Multiply by 1/x:    {time_mul_py:.4f}s")

# print(f"  → Speedup: {time_div_py / time_mul_py:.2f}x\n")

# time_div_np = timeit.timeit(divide_numpy, number=iterations)
# print(f"NumPy array - Direct division:    {time_div_np:.4f}s")

# time_mul_np = timeit.timeit(multiply_numpy, number=iterations)
# print(f"NumPy array - Multiply by 1/x:   {time_mul_np:.4f}s")

# print(f"  → Speedup: {time_div_np / time_mul_np:.2f}x")

from itertools import chain

nested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]] * 100

# List comprehension
print(timeit.timeit(lambda: [el for l in nested for el in l], number=10000))

# chain
print(timeit.timeit(lambda: list(chain.from_iterable(nested)), number=10000))