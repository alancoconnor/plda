import unittest
import PLDA
import numpy as np
from numpy.random import multivariate_normal as m_normal
from scipy.linalg import det
from scipy.linalg import eig

class TestPLDA(unittest.TestCase):
    def setUp(self):
        self.K = 10
        self.n_dims = 2

        self.μ_list = [np.ones(self.n_dims) * x * 10 for x in range(self.K)]
        self.n_list = [100 * (x % 2 + 1) for x in range(self.K)]
        self.w_cov = np.eye(self.n_dims) * np.random.randint(-10, 10,
                                                             self.n_dims)
        self.w_cov = np.matmul(self.w_cov, self.w_cov.T)

        self.X = [m_normal(self.μ_list[x], self.w_cov, self.n_list[x]) \
                  for x in range(self.K)]

        self.X = np.vstack(self.X)
        self.Y = []
        for x in range(len(self.n_list)):
            self.Y += [x] * self.n_list[x]
        self.Y = np.array(self.Y)
        data = [(x, y) for (x, y) in zip(self.X, self.Y)]
        self.model = PLDA.PLDA(data)

    def test_dummy(self):
        print(self.X.shape)
        print(self.Y.shape)

    def test_data_list_to_data_dict(self):
        idxs = np.cumsum(np.array(self.n_list)) - self.n_list[0]
        labels = list(self.model.data.keys())
        for x in range(len(labels)):
            label = labels[x]
            X = self.X[idxs[x]: idxs[x] + self.n_list[x], :]
            result = np.array(self.model.data[label])

            self.assertTrue(np.array_equal(result, X))

    def get_params_data_structure(self):
        ds = self.model.get_params_data_structure()
        n_keys = len(list(ds.keys())) 
        
        self.assertEqual(n_keys, 10 + self.K)

    def test_calc_A(self):
        self.assert_invertible(self.model.A)
        # inv( [A][(n / (n - 1) * Λ_w) ** (-.5)] ).T = W; See p. 537.
        result = self.model.Λ_w * self.model.n_avg / (self.model.n_avg - 1)
        result = result ** .5
        result = 1 / result
        result = self.model.A * result  # Since result should be diagonal.
        result = np.linalg.inv(result).T

        self.assertTrue(np.array_equal(result, self.model.W))

    def test_calc_class_log_probs(self):
        pass

    def test_calc_K(self):
        K = self.model.calc_K()
        self.assertEqual(K, self.K)

    def test_calc_Λ_b(self):
        self.assert_diagonal(self.model.Λ_b)

        inv_W = np.linalg.inv(self.model.W)
        result = np.matmul(inv_W.T, self.model.Λ_b)
        result = np.matmul(result, self.model.W.T)

        self.assertTrue(np.array_equal(result, self.model.S_b))

    def test_calc_Λ_w(self):
        self.assert_diagonal(self.model.Λ_w)

        inv_W = np.linalg.inv(self.model.W)
        result = np.matmul(inv_W.T, self.model.Λ_w)
        result = np.matmul(result, self.model.W.T)

        self.assertTrue(np.array_equal(result, self.model.Λ_w))

    def test_calc_m(self):
        result = self.model.m
        expected = self.X.mean(axis=0)

        self.assertTrue(np.allclose(result, expected, rtol=.0000000001))

    def test_calc_N(self):
        result = self.model.N
        expected = np.array(self.n_list).sum()

        self.assertEqual(result, expected)

    def test_calc_n_avg(self):
        n_avg = np.array(self.n_list).mean()

        self.assertEqual(self.model.n_avg, n_avg)

    def test_calc_Ψ(self):
        """ Verify Ψ using p. 537, Fig. 2. NOTE: n is approximated as n_avg.
            max(0, (n_avg - 1) / n_avg * (Λ_b / Λ_w) - 1 / n_avg) """
        self.assert_diagonal(Ψ)
        n = self.model.n_avg

        # Recall that Λ_b, Λ_w, and Ψ are all diagonal matrices.
        diag_b = self.model.Λ_b.diagonal()
        diag_w = self.model.Λ_w.diagonal()
        with np.errstate(divide='ignore', invalid='ignore'):
            Ψ = (n - 1) / n * (diag_b / diag_w) - (1 / n)
        Ψ[np.isnan(Ψ)] = 0
        Ψ = np.abs(Ψ)
        Ψ[np.isinf(Ψ)] = 0
        Ψ = np.diag(Ψ)

        self.assertTrue(np.array_equal(self.model.Ψ, Ψ))

    def test_calc_S_b(self):
        """ Verify S_b using Equation (1) on p. 532, section 2. """
        # Assert that m is correctly computed (p. 537, Fig. 2).
        N = np.array(self.n_list).sum()
        means = np.array(self.model.get_μs())
        weights = np.array(self.n_list) / N
        m = (means.T * weights).T

        self.assertTrue(np.array_equal(self.model.m, m))

        # Compute S_b.
        means = np.array(self.model.get_μs())
        diffs = means - self.model.m
        S_b = np.matmul(diffs.T * weights, diffs)

        self.assertTrue(np.array_equal(self.model.S_b, S_b))

    def test_calc_S_w(self):
        """ Veriy S_w using equation (1) (p. 532, section 2). """
        result = self.model.calc_S_w()

        S_w = [] # List of within-class scatters for all classes.
        for label in self.model.data.keys():
            mean = self.model.stats[label]['mean']
            N = self.K * self.class_sample_size
            data = np.array(self.model.data[label])

            # Assert that the class mean is computed correctly.
            is_same = np.array_equal(data.mean(axis=0), mean)
            self.assertTrue(is_same)

            s_w = data - mean
            s_w = np.matmul(s_w.T, s_w)  # Scatter within a class.
            S_w.append(s_w)
        S_w = np.array(S_w)
        S_w = S_w.sum(axis=0)
        S_w /= N  # Weighted-mean of the within-class scatters.

        result = self.model.S_w

        are_same = np.array_equal(result, S_w)
        self.assertTrue(are_same)            
        
        result = self.model.params['S_w']
        self.assertTrue(np.array_equal(result, expected))
               
    def test_calc_W(self):
        self.assert_invertible(self.model.W)

        vals, W = eig(self.model.S_b, self.model.S_w)
        self.assertTrue(np.array_equal(W, self.model.W))
        self.assertTrue(np.array_equal(self.model.W, self.model.W.T))

    def test_get_covariances(self):
        """ Verifies that returned covariacnes are correct and in order. """
        covs = self.model.get_covariances()

        labels = list(self.model.stats.keys())
        for x in range(len(labels)):
            label = labels[x] 
            result = covs[x]
            expected = self.model.stats[label]['covariance']
            self.assertTrue(np.array_equal(result, expected))

    def test_get_μs(self):
        """ Verifies that returned means are correct and in order. """
        μs = self.model.get_μs()

        labels = list(self.model.stats.keys())
        for x in range(len(labels)):
            label = labels[x] 
            result = μs[x]
            expected = self.model.stats[label]['μ']
            self.assertTrue(np.array_equal(result, expected))

    def test_get_sample_sizes(self):
        """ Verifies that returned sample sizes are correct and in order. """
        ns = self.model.get_sample_sizes()

        labels = list(self.model.stats.keys())
        for x in range(len(labels)):
            label = labels[x]
            result = ns[x]
            expected = self.model.stats[label]['n']

            self.assertEqual(result, expected)

    def test_get_stats_data_structure(self):
        result = self.model.get_params_data_structure()
        self.assertIsInstance(result, dict)

        expected = {'mean': None,
                    'n': None,
                    'covariance': None}

        self.assertEqual(result, expected)

    def test_set_params(self):
        # Need to make sure the model's parameter attributes are stored
        #  properly in the params data structure.
        pass
    def test_set_pdfs(self):
        pass
    def test_set_stats(self):
        pass
    def test_add_datum(self):
        pass
    def test_fit(self):
        pass
    def test_equals(self):
        pass
    def test_predict_class(self):
        pass
    def test_to_data_list(self):
        pass
    def test_update_stats(self):
        pass

    def test_whiten(self):
        U = self.X - self.model.m
        inv_A = np.linalg.inv(self.model.A)
        U = np.matmul(inv_A, U.T).T

        # u = inv(A)(x - m)
        result = self.model.whiten(self.X)
        self.assertTrue(np.array_equal(result, U))

        # x = [A]u + m
        result = np.matmul(self.model.A, result.T).T + self.model.m
        self.assertTrue(np.array_equal(result, self.X))
        
    def assert_diagonal(self, A, tolerance=None):
        """ Tolerance is the number of decimals to round at. """
        diagonal = A.diagonal()
        if tolerance is not None:
            is_diag = np.array_equal(np.around(A, tolerance),
                                     np.around(np.diag(diagonal), tolerance))
        else:
            is_diag = np.array_equal(A, np.diag(diagonal))

        self.assertTrue(is_diag)

    def assert_invertible(self, A):
        determinant = det(A)
        is_invertible = determinant != 0

        self.assertTrue(is_invertible)