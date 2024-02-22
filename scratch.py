import numpy as np

A = np.array([1, 2])
B = np.array([3, 4])

C = np.vstack((A, B))
print(C[:, 0])