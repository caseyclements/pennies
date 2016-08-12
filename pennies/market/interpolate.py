"""Curve Interpolators"""
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.linalg import solve_banded, solve

# !!! TODO None of this is necessary! scipy.interpolate does it !!!


class CurveInterpolator(object):
    def __init__(self):
        pass


class NodeInterpolator(CurveInterpolator):
    """Interpolation on curves defined by node points.

    Interpolated value, y(x) is linear in node values y_i(x_i),
    i.e. y(x) = sum_i a_i(x) * y_i(x_i).
    Weights depend on Interpolation Scheme, and must be calibrated.
    """
    def __init__(self, x, y):
        super(NodeInterpolator, self).__init__()
        self.x = x
        self.y = y
        self.n_nodes = len(x)
        if self.n_nodes != len(y):
            raise ValueError("Length of x ({}) differs from length of y ({})"
                             .format(self.n_nodes, len(y)))

    def __call__(self, x, *args, **kwargs):
        return (self.weights(x) * self.y).sum()

    def weights(self, x):
        raise NotImplementedError("NodeInterpolator is an abstract base class.")


class NodeInterpolatorDummy(NodeInterpolator):
    """NodeInterpolator with fixed weights. FOR TESTING ONLY."""
    def __init__(self, x, y):
        super(NodeInterpolatorDummy, self).__init__(x, y)

    def weights(self, x):
        return 1 / self.n_nodes


class CubicSpline(object):
    """Piecewise cubic polynomial representation of a curve.

    For our representation, see
    http://www.geos.ed.ac.uk/~yliu23/docs/lect_spline.pdf

    f(x) = a(x) * y_k + b(x) * y_{k+1} + c(x) * y''_k + d(x) * y''_{k+1}

    a = (x_{k+1} - x) / (x_{k+1} - x_{k})
    b = 1 - a = (x - x_k) / (x_{k+1} - x_{k})
    c = (a^3 - a) * (x_{k+1} - x_k)^2 / 6
    d = (b^3 - b) * (x_{k+1} - x_k)^2 / 6

    This is used in place of scipy.interpolate.CubicSpline
    (a) to compute node sensitivities
    (b) for testing against scipy
    """

    def __init__(self, x, y, order=3):

        self.x = x
        self.y = y
        self.d2y_dx2 = np.zeros_like(y)  # Piecewise linear case
        n = len(x)
        self.n = n
        assert len(y) == n

        if order < 3:
            return

        # Set up a tridiagonal system to solve for y''
        A = np.zeros((3, n))
        dx = np.diff(x)
        slope = np.diff(y) / dx
        # Define n-2 elements
        A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # diagonal
        A[0, :-2] = dx[1:]   # upper diagonal
        A[-1, 2:] = dx[:-1]  # lower diagonal
        # Apply natural boundaries ==> y''[0]==y''[-1] = 0
        A[1, 0] = A[1, -1] = 1

        b = np.zeros_like(y)  # len = n
        b[1:-1] = 6 * (slope[1:] - slope[:-1])  # len(b) == n - 2

        self.d2y_dx2 = solve_banded((1, 1), A, b, overwrite_ab=False,
                                    overwrite_b=False, check_finite=False)

        #------Manually-----------

        AA = np.zeros((n,n))
        diag = np.ones((n,))
        diag[1:-1] = 2 * (dx[:-1] + dx[1:])
        AA[np.diag_indices(n)] = diag
        for i in range(1, n-1):
            AA[i, i - 1] = dx[i-1]
            AA[i, i + 1] = dx[i]
        print('AA')
        print(AA)
        print('b')
        print(b)
        ypp = solve(AA, b)

        print('solve_banded = {}'.format(self.d2y_dx2))
        print('solve = {}'.format(ypp))
        print(self.d2y_dx2 - ypp)
        print('INIT COMPLETE')
        self.d2y_dx2 = ypp

    def a(self, x):
        i = np.digitize(x, self.x)
        assert 0 < i < self.n, 'Currently not doing extrapolation'
        return (self.x[i] - x) / (self.x[i] - self.x[i-1])

    def b(self, x):
        return 1 - self.a(x)

    def c(self, x):
        i = np.digitize(x, self.x)
        assert 0 < i < self.n, 'Currently not doing extrapolation'
        a = self.a(x)
        return (a**3 - a) * (x[i] - x[i-1])**2 / 6

    def c(self, x):
        i = np.digitize(x, self.x)
        assert 0 < i < self.n, 'Currently not doing extrapolation'
        b = self.b(x)
        return (b ** 3 - b) * (x[i] - x[i - 1]) ** 2 / 6

    def __call__(self, x):
        i = np.digitize(x, self.x)
        #assert np.all(0 < i) and np.all(i < self.n-1), 'Currently not doing extrapolation'
        dx = np.diff(self.x)
        a = (self.x[i] - x) / dx[i-1]
        b = 1 - a
        c = (a**3 - a) * dx[i-1]**2 / 6
        d = (a**3 - a) * dx[i-1]**2 / 6
        return (a * self.y[i-1] +
                b * self.y[i] +
                c * self.d2y_dx2[i-1] +
                d * self.d2y_dx2[i])


if __name__ == '__main__':

    n_nodes = 5
    x = np.arange(5)
    def f(x):
        return np.sin(x)
    y = f(x)
    pc_cube = CubicSpline(x, y)
    pc_lin = CubicSpline(x, y, order=1)

    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, y, bc_type='natural')


    # TEST SINGLE POINT
    x_test = 2.2
    print('f({}) = {}'.format(x_test, f(x_test)))
    print('linear est of f({}) = {}'.format(x_test, pc_lin(x_test)))
    print('cubic est of f({}) = {}'.format(x_test, pc_cube(x_test)))
    print('scipy cspline est of f({}) = {}'.format(x_test, cs(x_test)))

    # PLOTS
    xs = np.linspace(0, x[-1] * 49. / 50.)
    ys_lin = pc_lin(xs)
    ys_cube = pc_cube(xs)
    ys_cs = cs(xs)

    test_a = np.vstack((xs,ys_cs, ys_cube, ys_lin))
    print(test_a)



    import matplotlib.pyplot as plt
    plt.figure(figsize=(24,12))
    plt.plot(x, y, 'o', label='data')
    plt.plot(xs, np.sin(xs), label='true')
    plt.plot(xs, ys_lin, label="pc_linear")
    plt.plot(xs, ys_cube, label="pc_cubic")
    plt.plot(xs, ys_cs, label="scipy")
    plt.legend(loc='lower left', ncol=2)
    print('FIN')






