import numpy as np
import time
import torch



# Create two square matrices with N elements using NumPy
N = 10000

inicio = time.time()
A = np.random.rand(N, N)
B = np.random.rand(N, N)
# Multiply the matrices using NumPy
C = np.dot(A, B)
print("Result of matrix multiplication using NumPy:")
print(C)
fin = time.time()
print("Tiempo de ejecución numpy: ", fin - inicio)

inicio = time.time()
A = torch.rand(N, N)
B = torch.rand(N, N)
# Multiply the matrices using PyTorch
C = torch.matmul(A, B)
print("Result of matrix multiplication using PyTorch:")
print(C)
fin = time.time()
print("Tiempo de ejecución pytorch: ", fin - inicio)

