"""Script to demonstrate basic concepts of LDA/PLDA.

Some examples to illustrate the very basics.

Start from LDA, as described in Duda, Hart & Stork textbook.
Expand to PLDA, with classes unseen in training.
Validate results of simple implementation against library.

Copyright (c) 2024 Alan C. O'Connor
"""

import numpy as np
import matplotlib.pyplot as plt

## Example 1. Two classes, Gaussian distribution.
# Compute the Fisher linear discriminant. plot.

ndim = 2
nsamp = 200*ndim

# true means, variance
mu0 = np.random.randn(ndim)
mu1 = np.random.randn(ndim)
Sigma = np.array([[1.0, 0.2], [0.2, 0.3]])

# generate data
U, S, V = np.linalg.svd(Sigma)
rootSigma = U @ np.diag(np.sqrt(S))
assert np.allclose(rootSigma @ rootSigma.T, Sigma)

x0 = mu0[:, np.newaxis] + rootSigma @ np.random.randn(ndim, nsamp)
x1 = mu1[:, np.newaxis] + rootSigma @ np.random.randn(ndim, nsamp)

# sample mean and scatter matrices
m0 = np.mean(x0, axis=1)
m1 = np.mean(x1, axis=1)
d0 = x0 - m0[:, np.newaxis]
d1 = x1 - m1[:, np.newaxis]
S0 = np.sum(d0[:, np.newaxis, :] * d0[np.newaxis, :, :], axis=2)
S1 = np.sum(d1[:, np.newaxis, :] * d1[np.newaxis, :, :], axis=2)
Sw = S0 + S1 # within-class scatter matrix
Sb = (m1 - m0)[:, np.newaxis] * (m1 - m0)[np.newaxis, :] # between-class scatter

w_lda = np.linalg.solve(Sw, (m1-m0))

print(w_lda)
print(w_lda[:, np.newaxis].T @ x0)
plt.figure()
plt.hist(np.ravel(w_lda[:, np.newaxis].T @ x0), bins=20)
plt.hist(np.ravel(w_lda[:, np.newaxis].T @ x1), bins=20)

plt.figure()
plt.scatter(x0[0, :], x0[1, :])
plt.scatter(x1[0, :], x1[1, :])
# plt.show()

# Example 2. Multiple discriminant analysis, higher dimensions, with homoskedastic Guassian distribution
# See Duda, Hart, and Stork section 3.8.3
# d dimensions
# c classes
# d >= c

nclass = 5
ndim = 10
nsamp = 10*ndim

# true means for each class
mu = np.random.randn(ndim, nclass)

# true variance
rootSigma = np.diag(0.1 + 0.9 * np.random.rand(ndim)) @ np.random.randn(ndim, nsamp)
Sigma = 1.0/nsamp * rootSigma @ rootSigma.T
U, S, V = np.linalg.svd(Sigma)
rootSigma = U @ np.diag(np.sqrt(S))
assert np.allclose(rootSigma @ rootSigma.T, Sigma)

# generate balanced dataset for all classes simultaneously
x = mu[:, :, np.newaxis] + np.einsum('ij,jk...->ik...', rootSigma, np.random.randn(ndim, nclass, nsamp))

# sample mean and per-class scatter matrices
m = np.mean(x, axis=2)
d = x - m[:, :, np.newaxis]
S = np.sum(d[:, np.newaxis, :, :] * d[np.newaxis, :, :, :], axis=3)

# total within-class scatter matrix
Sw = np.sum(S, axis=2) 

# in general case, need to account for the difference in sample numbers between classes
total_mean = np.mean(m, axis=1)

# total between-class scatter matrix
d_between = (m - total_mean[:, np.newaxis])
Sb = d_between @ d_between.T

# find the transformation that maximizes ratio of determinant of between scatter to determinant of within scatter
# solve generalize eigenvalue problem: Sb v - lambda Sw v = 0
from scipy.linalg import eig as generalized_eigs
W_eigvals, W_eigvecs = generalized_eigs(Sb, Sw)
W_eigvals = np.real(W_eigvals)   # real anyway since both matrices are pos def
W_eigvecs = np.real(W_eigvecs)

# scipy does not sort the eigenvalues for us
eigsortargs = np.argsort(W_eigvals)
W_eigvals_sort = W_eigvals[eigsortargs[::-1]]
W_eigvecs_sort = W_eigvecs[:, eigsortargs[::-1]]

# expect (nclass - 1) nonzero eigenvalues
W = W_eigvecs_sort[:, :(nclass-1)]

# transform data with W.
x_trans = np.einsum('ij,ik...->jk...', W, x)

plt.figure()
plt.plot(W_eigvals, label='unsorted')
plt.plot(W_eigvals_sort, label='sorted')
plt.legend()

plt.figure()
for c in range(nclass):
    plt.scatter(x[0, c, :], x[1, c, :])
plt.title(f'Before transformation (first 2 of {ndim} dims shown)')

plt.figure()
for c in range(nclass):
    plt.scatter(x_trans[0, c, :], x_trans[1, c, :])
plt.title(f'After transformation (first 2 of {nclass-1} dims shown)')
plt.show()

breakpoint()
