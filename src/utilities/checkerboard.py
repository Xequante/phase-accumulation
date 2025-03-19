import numpy as np

def checkerboard(n, m, pn, pm):
    checkerboard = np.zeros((n, m))
    for i in range(n):
        if i % pn < pn / 2:
            checkerboard[i] += 1
    for i in range(m):
        if i % pm < pm / 2:
            checkerboard[:, i] += 1
    return np.mod(checkerboard, 2)
