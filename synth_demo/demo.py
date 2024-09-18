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
if False:
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

nclass = 20
nclass_train = nclass // 2 # withold some classes from training for PLDA
ndim = 6
nsamp = 50
nsamp_train = 6

if True:
    # randomized case

    # true means for each class
    mu = np.random.randn(ndim, nclass)

    # true variance
    rootSigma = np.diag(0.1 + 0.7 * np.random.rand(ndim)) @ np.random.randn(ndim, nsamp)
    Sigma = 1.0/nsamp * rootSigma @ rootSigma.T
    U, S, V = np.linalg.svd(Sigma)
    rootSigma = U @ np.diag(np.sqrt(S))
    assert np.allclose(rootSigma @ rootSigma.T, Sigma)
else:
    # simple test case where all between-class variation is in the first dimension

    # true means for each class only differ in the first dimension
    mu = np.vstack((np.random.randn(1, nclass), np.zeros((ndim-1, nclass))))

    # true variance
    rootSigma = np.diag(0.1 * np.ones((ndim))) @ np.random.randn(ndim, nsamp)
    Sigma = 1.0/nsamp * rootSigma @ rootSigma.T
    U, S, V = np.linalg.svd(Sigma)
    rootSigma = U @ np.diag(np.sqrt(S))
    assert np.allclose(rootSigma @ rootSigma.T, Sigma)

# generate balanced dataset for all classes simultaneously
x = mu[:, :, np.newaxis] + np.einsum('ij,jk...->ik...', rootSigma, np.random.randn(ndim, nclass, nsamp))

# sample mean and per-class scatter matrices
m = np.mean(x[:, :, :nsamp_train], axis=2)
d = x[:, :, :nsamp_train] - m[:, :, np.newaxis]
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
    plt.plot(m[0, c], m[1, c], 'ks')
plt.title(f'Before transformation (first 2 of {ndim} dims shown)')

plt.figure()
for c in range(nclass):
    plt.scatter(x_trans[0, c, :], x_trans[1, c, :])
    plt.plot(np.mean(x_trans[0, c, :]), np.mean(x_trans[1, c, :]), 'ks')
plt.title(f'After LDA transformation (first 2 of {nclass-1} dims shown)')
# plt.show()


## Now on to PLDA.


import plda 

# arrange data into single set with corresonding labels

# x above has shape (ndim, nclass, nsamp_per_class)
# plda class expects (nsamp, ndim)

x_train = np.reshape(x[:, :nclass_train, :nsamp_train], (ndim, nclass_train * nsamp_train))
x_train = x_train.transpose()

x_test = np.reshape(x, (ndim, nclass * nsamp))
x_test = x_test.transpose()

training_labels = np.arange(0, nclass_train)[:, np.newaxis] * np.ones((1, nsamp_train))
training_labels = np.reshape(training_labels, (nclass_train * nsamp_train, ))

test_labels = np.arange(0, nclass)[:, np.newaxis] * np.ones((1, nsamp))
test_labels = np.reshape(test_labels, (nclass * nsamp, ))

# Use as many principal components in the data as possible.
plda_model = plda.Classifier()
plda_model.fit_model(x_train, training_labels, n_principal_components=ndim)


print(plda_model.model.prior_params['cov_diag'])
print(plda_model.model.posterior_params[0]['cov_diag'])

U_model = plda_model.model.transform(x_test, from_space='D', to_space='U_model')

U_model_by_class = np.reshape(U_model.transpose(), (-1, nclass, nsamp))
transformed_sample_means = np.mean(U_model_by_class[:,:,:nsamp_train], axis=2)  # (ndim, nclass)
classwise_centers = np.array([plda_model.model.posterior_params[c]['mean'] for c in range(nclass_train)])

plt.figure()
for c in range(nclass):
    plt.scatter(U_model_by_class[-1, c, :], U_model_by_class[-2, c, :])
plt.scatter(transformed_sample_means[-1, :], transformed_sample_means[-2, :], c='k', marker='s', label='Sample Means')
plt.scatter(classwise_centers[:, -1], classwise_centers[:, -2], c='c', marker='s', label='Model Centers')
plt.legend()
plt.title(f'After PLDA transformation (last 2 of {ndim} dims shown)\n Model centers fit by EM are close to sample means')

# breakpoint()

if (nclass * nsamp) < 200:
    # comparison of each example (including classes not seen during fit)

    log_ratios_grid = np.zeros((nclass * nsamp, nclass * nsamp))
    for ii in range(0, nclass * nsamp):
        for jj in range(0, nclass * nsamp):
            log_ratios_grid[ii, jj] = plda_model.model.calc_same_diff_log_likelihood_ratio(U_model[ii][None,], U_model[jj][None,])
            # if jj > ii:
            #     log_ratios_grid[jj, ii] = 
    plt.figure()
    plt.imshow(log_ratios_grid)
    plt.clim(-4, 4)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.title("Same / Diff Log Likelihood Ratio between Samples\n(higher = more similar)")

else:
    # too many
    print('Skipping exhaustive computation of log ratios')

# draw pairs of same and different classes (including classes not see during fit),
# and observe distribution of log likelihood ratios
log_ratios_same = []
log_ratios_diff = []

for ii in range(0, 10000):
    ind_0 = np.random.randint(nclass * nsamp)
    ind_1 = np.random.randint(nclass * nsamp)

    log_ratio = plda_model.model.calc_same_diff_log_likelihood_ratio(U_model[ind_0][None,], U_model[ind_1][None,])
    if test_labels[ind_0] == test_labels[ind_1]:
        log_ratios_same.append(log_ratio)
    else:
        log_ratios_diff.append(log_ratio)

log_ratios_diff = np.array(log_ratios_diff)
log_ratios_same = np.array(log_ratios_same)


# get ROC: probability of correct or false detection as decision threshold is varied
threshvals = np.linspace(-2, 5, num=1000)
pfa =  np.zeros_like(threshvals)
pd = np.zeros_like(threshvals)
for ii, thresh in enumerate(threshvals):
    pd[ii] = np.sum(log_ratios_same>thresh) / len(log_ratios_same)
    pfa[ii] = np.sum(log_ratios_diff>thresh) / len(log_ratios_diff)

plt.figure()
plt.plot(np.sort(log_ratios_same), np.linspace(0, 1, len(log_ratios_same)), label='Same Class')
plt.plot(np.sort(log_ratios_diff), np.linspace(0, 1, len(log_ratios_diff)), label='Different Class')
plt.legend()
plt.xlabel('Log Likelihood Ratio')
plt.ylabel('Empirical CDF')
plt.xlim(-5, 5)

plt.figure(10)
plt.plot(pfa, pd, label="ROC for Comparing Against a single Enrollment Vector")
plt.title('ROCs')
plt.xlabel('P_FA = Different Class incorrectly labeled same')
plt.ylabel('P_D = Same class correctly detected')

plt.figure(11)
plt.loglog(pfa, 1-pd, label="ROC for Comparing Against a single Enrollment Vector")
plt.title('ROCs')
plt.xlabel('log10(False Positive %)')
plt.ylabel('log10(False Negative %)')

# get a second ROC where test vectors are compared to the class means
# TODO. this is not right yet. Ioffe gives the right formulae for posterior probabilities
# for same / different for a multi-example "gallery". Need to use those.
log_ratios_same_mean = []
log_ratios_diff_mean = []

for ii in range(0, 10000):
    ind_test_vec = np.random.randint(nclass * nsamp)
    ind_class = np.random.randint(nclass)

    log_ratio = plda_model.model.calc_same_diff_log_likelihood_ratio(U_model[ind_test_vec][None,], transformed_sample_means[:, ind_class][None,])
    if test_labels[ind_test_vec] == ind_class:
        log_ratios_same_mean.append(log_ratio)
    else:
        log_ratios_diff_mean.append(log_ratio)

log_ratios_diff_mean = np.array(log_ratios_diff_mean)
log_ratios_same_mean = np.array(log_ratios_same_mean)


# get ROC: probability of correct or false detection as decision threshold is varied
threshvals = np.linspace(-2, 5, num=1000)
pfa_mean =  np.zeros_like(threshvals)
pd_mean = np.zeros_like(threshvals)
for ii, thresh in enumerate(threshvals):
    pd_mean[ii] = np.sum(log_ratios_same_mean>thresh) / len(log_ratios_same_mean)
    pfa_mean[ii] = np.sum(log_ratios_diff_mean>thresh) / len(log_ratios_diff_mean)

# plt.figure()
# plt.plot(np.sort(log_ratios_same), np.linspace(0, 1, len(log_ratios_same)), label='Same Class')
# plt.plot(np.sort(log_ratios_diff), np.linspace(0, 1, len(log_ratios_diff)), label='Different Class')
# plt.legend()
# plt.xlabel('Log Likelihood Ratio')
# plt.ylabel('Empirical CDF')
# plt.xlim(-5, 5)

plt.figure(10)
plt.plot(pfa_mean, pd_mean, label='ROC for Comparing Against Mean of Enrollment Gallery')
plt.figure(11)
plt.loglog(pfa_mean, 1-pd_mean, label='ROC for Comparing Against Mean of Enrollment Gallery')


# get a third ROC where test vectors are compared to the full multi-example "gallery"
log_ratios_same_mean = []
log_ratios_diff_mean = []

for ii in range(0, 10000):
    ind_test_vec = np.random.randint(nclass * nsamp)
    ind_class = np.random.randint(nclass)
    inds_gallery = range(ind_class*nsamp, (ind_class*nsamp+nsamp_train))

    log_ratio = plda_model.model.calc_same_diff_log_likelihood_ratio(U_model[ind_test_vec][None,], U_model_g=U_model[inds_gallery])
    if test_labels[ind_test_vec] == ind_class:
        log_ratios_same_mean.append(log_ratio)
    else:
        log_ratios_diff_mean.append(log_ratio)

log_ratios_diff_mean = np.array(log_ratios_diff_mean)
log_ratios_same_mean = np.array(log_ratios_same_mean)


# get ROC: probability of correct or false detection as decision threshold is varied
threshvals = np.linspace(-2, 5, num=1000)
pfa_mean =  np.zeros_like(threshvals)
pd_mean = np.zeros_like(threshvals)
for ii, thresh in enumerate(threshvals):
    pd_mean[ii] = np.sum(log_ratios_same_mean>thresh) / len(log_ratios_same_mean)
    pfa_mean[ii] = np.sum(log_ratios_diff_mean>thresh) / len(log_ratios_diff_mean)

# plt.figure()
# plt.plot(np.sort(log_ratios_same), np.linspace(0, 1, len(log_ratios_same)), label='Same Class')
# plt.plot(np.sort(log_ratios_diff), np.linspace(0, 1, len(log_ratios_diff)), label='Different Class')
# plt.legend()
# plt.xlabel('Log Likelihood Ratio')
# plt.ylabel('Empirical CDF')
# plt.xlim(-5, 5)

plt.figure(10)
plt.plot(pfa_mean, pd_mean, label='ROC for Comparing Against Multiexample Enrollment')
plt.legend()

plt.figure(11)
plt.loglog(pfa_mean, 1-pd_mean, label='ROC for Comparing Against Mean of Enrollment Gallery')
plt.legend()
plt.grid(True)

plt.show()

## TODO. my own implementation of PLDA for the "two-covariance" flavor
# Model is that there is a prior on class centers: p(y) ~ N(m, Phi_b) and 
# conditional probability of data given class id: p(x|y) ~ N(y, Phi_w )
# Phi_b and Phi_w are related to between-class and within class scatter matrices of LDA.
# Use the same transformation derived from generalize eigenvector problem above
# to simultaneously diagonalize Phi_b and Phi_w.

