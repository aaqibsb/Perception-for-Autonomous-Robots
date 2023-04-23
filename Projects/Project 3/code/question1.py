import numpy as np
from scipy import linalg

""" World points """
X = [(0, 0, 0), (0, 3, 0), (0, 7, 0), (0, 11, 0),
     (7, 1, 0), (0, 11, 7), (7, 9, 0), (0, 1, 7)]

""" Image points """
x = [(757, 213), (758, 415), (758, 686), (759, 966),
     (1190, 172), (329, 1041), (1204, 850), (340, 159)]

A = np.empty((0, 12))

""" Setting up A matrix """
for i in range(0, len(X)):
    u = x[i][0]
    v = x[i][1]
    w = 1

    Xi = X[i][0]
    Yi = X[i][1]
    Zi = X[i][2]

    A1 = np.matrix([[0, 0, 0, 0, -w * Xi, -w * Yi, -w * Zi, -w * 1, v * Xi, v * Yi, v * Zi, v * 1]])
    A2 = np.matrix([[w * Xi, w * Yi, w * Zi, w * 1, 0, 0, 0, 0, -u * Xi, -u * Yi, -u * Zi, -u * 1]])
    A3 = np.matrix([[-v * Xi, -v * Yi, -v * Zi, -v * 1, u * Xi, u * Yi, u * Zi, u * 1, 0, 0, 0, 0]])

    A = np.vstack((A, A1, A2, A3))

# print("A:",'\n',A, sep='')

""" Projection Matrix (P) """
_, _, v = np.linalg.svd(A)
P = np.reshape(v[-1],(3,4))
P = P/P[-1,-1]
print("P:",'\n', P, sep='')
print()

""" Translation Matrix (C) """
_, _, V = np.linalg.svd(P)
C = V[-1]
C = C/C[0,-1]
C = np.reshape(C,(4,1))
print("C:",'\n', C, sep='')
print()

G = -C[:-1]
I = np.identity(3)
CT = np.hstack((I, G))
print("CT:",'\n', CT, sep='')
print()

""" SubMatrix (M) """
M = P @ np.linalg.pinv(CT)
print("M:",'\n', M, sep='')
print()

""" RQ Factorization of M """
K, R = linalg.rq(M)

""" Rotation Matrix (R) """
print("R:",'\n', R, sep='')
print()

""" Intrinsic Matrix (K) """
print("K:",'\n', K, sep='')
print()

""" Calculating Re-projection error """
X_interim = np.asarray(X)
X_pad = np.pad(X_interim, ((0, 0), (0, 1)), mode='constant', constant_values=1)

X_new = P @ X_pad.T
X_new = X_new/X_new[2]
X_new = X_new.T[:, :2]
error = np.linalg.norm(X_new - x, axis=1)
for i in range(len(error)):
    print('Re-projection error for point', X[i],":", np.round(error[i], 4))