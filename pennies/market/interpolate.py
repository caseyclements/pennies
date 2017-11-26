"""Curve Interpolators"""
from __future__ import absolute_import, division, print_function

import numpy as np

from scipy.interpolate import PPoly, CubicSpline, interp1d
from scipy.linalg import solve_banded, solve
from numpy import repeat, prod, arange
from numpy.matlib import repmat
from six import string_types


class CurveInterpolator(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x, *args, **kwargs):
        raise NotImplementedError("CurveInterpolator is an abstract base class.")


class NodeInterpolator(CurveInterpolator):
    """Interpolation on curves defined by node points.

    Interpolated value, y(x) is linear in node values y_i(x_i),
    i.e. y(x) = sum_i a_i(x) * y_i(x_i).
    Weights depend on Interpolation Scheme, and must be calibrated.
    """
    def __init__(self, x, y, interpolator, *args, **kwargs):
        super(NodeInterpolator, self).__init__()
        self.x = x
        self.y = y
        self.n_nodes = len(x)
        if self.n_nodes != len(y):
            raise ValueError("Length of x ({}) differs from length of y ({})"
                             .format(self.n_nodes, len(y)))
        self._fn = interpolator(x,y, *args, **kwargs)

    def __call__(self, x, *args, **kwargs):
        self._fn(x, *args, **kwargs)


class PiecewiseLinear(CurveInterpolator):
    """Piecewise Linear NodeInterpolator.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-d array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along `axis` (see below)
        must match the length of `x`. Values must be finite.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    extrapolate : 2-tuple, optional
        Condition to be applied above and below bounds of x.

        The first value applies to values below x[0].
        The second value applies to values above x[-1].

        * None: No extrapolation. NaN is returned.
        * 'clamped': The first derivative at curves ends are zero.
         Equivalent to a value of 0.0.
        * 'natural': The derivative beyond the last point is equal to the
         derivative of the closest interval.
        * A number: derivative outside bound.

    Attributes
    ----------
    x : ndarray, shape (n,)
        Breakpoints. The same `x` which was passed to the constructor.
    y: ndarray, shape (n, ...)

    Methods
    -------
    __call__
    node_derivative

    See Also
    --------
    CubicSplineWithNodeSens, scipy.interpolate

    Examples
    --------
    In this example, we use PiecewiseLinear to interpolate a sampled sinusoid.

    >>> from pennies.market.interpolate import PiecewiseLinear
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(10)
    >>> y = np.sin(x)
    >>> interp = PiecewiseLinear(x, y)
    >>> xs = np.arange(-0.5, 9.6, 0.1)
    >>> p = plt.figure(figsize=(6.5, 4))
    >>> p = plt.plot(x, y, 'o', label='data')
    >>> p = plt.plot(xs, np.sin(xs), label='true')
    >>> p = plt.plot(xs, interp(xs), label='piecewise linear')
    >>> p = plt.legend(loc='lower left', ncol=3)
    >>> p = plt.xlim(-0.5, 9.5)
    >>> #plt.show()

    """
    def __init__(self, x, y, extrapolate=('clamped', 'clamped')):
        super(PiecewiseLinear, self).__init__(x, y)
        self.x = x
        self.y = y
        self.n_nodes = len(x)
        if self.n_nodes != len(y):
            raise ValueError("Length of x ({}) differs from length of y ({})"
                             .format(self.n_nodes, len(y)))
        self.extrapolate = extrapolate
        # Store slopes (or recompute for each __call__?
        self.slopes = np.zeros((self.n_nodes + 1,) + y.shape[1:])
        # Interpolation.
        self.slopes[1:-1] = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        # Extrapolation: Low
        if self.extrapolate[0] == 'clamped' or self.extrapolate[0] == 0:
            self.slopes[0] = 0.0
        elif self.extrapolate[0] is None:
            self.slopes[0] = np.NaN
        elif self.extrapolate[0] == 'natural':
            self.slopes[0] = self.slopes[1]
        elif isinstance(self.extrapolate[0], np.number):
            self.slopes[0] = self.extrapolate[0]
        else:
            raise ValueError('1st member of extrapolate kwarg must be one of' +
                             'natural, clamped, None, or a value for the slope')
        # Extrapolation: High
        if self.extrapolate[1] == 'clamped' or self.extrapolate[1] == 0:
            self.slopes[-1] = 0.0
        elif self.extrapolate[1] is None:
            self.slopes[-1] = np.NaN
        elif self.extrapolate[1] == 'natural':
            self.slopes[-1] = self.slopes[-2]
        elif isinstance(self.extrapolate[1], np.number):
            self.slopes[-1] = self.extrapolate[1]
        else:
            raise ValueError('2nd member of extrapolate kwarg must be one of' +
                             'natural, clamped, None, or a value for the slope')

    def __call__(self, x, *args, **kwargs):
        """ Estimate y, given x."""
        i = np.digitize(x, self.x)
        return (self.y[np.maximum(i-1, 0)] +
                self.slopes[i] * (x - self.x[np.maximum(i-1, 0)]))

    def node_derivative(self, x):
        """Sensitivity of y(x) to a unit move in each node's y-value

        Parameters
        ----------
        x : array-like
            Points to evaluate the interpolant at.

        Returns
        -------
        dy(x)/dy_i : array-like
            Shape is determined by shape of y
            the interpolation axis in the original array with the shape of x.
        """
        x = np.asarray(x)
        ndim = x.ndim
        x_is_num = ndim == 0
        if x_is_num:
            x = np.array([x])
            ndim = 1

        idx = np.digitize(x, self.x)  # n+1 possible values
        # apply min/max to ensure out-of-bounds isn't triggered by interp
        # this also happens to produce the correct result for 'natural' extrap
        idx_safe = np.maximum(np.minimum(idx, self.n_nodes-1), 1)  # values in 1..n-1

        y_deriv = np.zeros(x.shape + (self.n_nodes,))
        inv_dx = 1.0 / (self.x[idx_safe] - self.x[idx_safe - 1])
        weight_right = ((x - self.x[idx_safe - 1]) * inv_dx).ravel()
        weight_left = ((self.x[idx_safe] - x) * inv_dx).ravel()
        # the following produces the indices in form [[x0], [x1],..[xlast]]
        # where x0 is an array of the indices to the 0'th axis of each point
        # http://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays
        indices = [repmat(repeat(range(x.shape[i]), prod(x.shape[i + 1:])),
                          1, int(prod(x.shape[:i])))
                   for i in range(ndim)]
        idx_flat = idx_safe.ravel()
        y_deriv[indices + [idx_flat]] = weight_right.ravel()
        y_deriv[indices + [idx_flat - 1]] = weight_left.ravel()

        # Extrapolation below
        if self.extrapolate[0] == 'natural':
            pass
        elif (self.extrapolate[0] == 'clamped' or
                  np.isscalar(self.extrapolate[0])):  # Slope is fixed
            extrap = idx == 0
            y_deriv[extrap, 0] = 1.
            y_deriv[extrap, 1:] = 0
        elif self.extrapolate[0] is None:
            y_deriv[idx == 0, ...] = np.NaN
        else:
            raise ValueError('1st member of extrapolate kwarg must be one' +
                             'natural, clamped, None, or a scalar slope value')

        # Extrapolation above
        if self.extrapolate[1] == 'natural':
            pass
        elif (self.extrapolate[1] == 'clamped' or
                  np.isscalar(self.extrapolate[0])):  # Slope is fixed
            extrap = idx == self.n_nodes
            y_deriv[extrap, -1] = 1.
            y_deriv[extrap, :-1] = 0
        elif self.extrapolate[1] is None:
            extrap = idx == self.n_nodes
            y_deriv[extrap, -1] = np.NaN
        else:
            raise ValueError('2nd member of extrapolate kwarg must be one' +
                             'natural, clamped, None, or a scalar slope value')

        if x_is_num:
            y_deriv = y_deriv[0,...]

        return y_deriv


class CubicSplineWithNodeSens(PPoly):
    """Cubic spline data interpolator.

    Interpolate data with a piecewise cubic polynomial which is twice
    continuously differentiable [1]_. The result is represented as a `PPoly`
    instance with breakpoints matching the given data.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-d array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along `axis` (see below)
        must match the length of `x`. Values must be finite.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    bc_type : string or 2-tuple, optional
        Boundary condition type. Two additional equations, given by the
        boundary conditions, are required to determine all coefficients of
        polynomials on each segment [2]_.

        If `bc_type` is a string, then the specified condition will be applied
        at both ends of a spline. Available conditions are:

        * 'not-a-knot' (default): The first and second segment at a curve end
          are the same polynomial. It is a good default when there is no
          information on boundary conditions.
        * 'periodic': The interpolated functions is assumed to be periodic
          of period ``x[-1] - x[0]``. The first and last value of `y` must be
          identical: ``y[0] == y[-1]``. This boundary condition will result in
          ``y'[0] == y'[-1]`` and ``y''[0] == y''[-1]``.
        * 'clamped': The first derivative at curves ends are zero. Assuming
          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
        * 'natural': The second derivative at curve ends are zero. Assuming
          a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.

        If `bc_type` is a 2-tuple, the first and the second value will be
        applied at the curve start and end respectively. The tuple values can
        be one of the previously mentioned strings (except 'periodic') or a
        tuple `(order, deriv_values)` allowing to specify arbitrary
        derivatives at curve ends:

        * `order`: the derivative order, 1 or 2.
        * `deriv_value`: array_like containing derivative values, shape must
          be the same as `y`, excluding `axis` dimension. For example, if `y`
          is 1D, then `deriv_value` must be a scalar. If `y` is 3D with the
          shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2D
          and have the shape (n0, n1).
    extrapolate : {bool, 'periodic', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. If None (default), `extrapolate` is
        set to 'periodic' for ``bc_type='periodic'`` and to True otherwise.

    Attributes
    ----------
    x : ndarray, shape (n,)
        Breakpoints. The same `x` which was passed to the constructor.
    c : ndarray, shape (4, n-1, ...)
        Coefficients of the polynomials on each segment. The trailing
        dimensions match the dimensions of `y`, excluding `axis`. For example,
        if `y` is 1-d, then ``c[k, i]`` is a coefficient for
        ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
    axis : int
        Interpolation axis. The same `axis` which was passed to the
        constructor.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    roots

    See Also
    --------
    Akima1DInterpolator
    PchipInterpolator
    PPoly

    Notes
    -----
    Parameters `bc_type` and `interpolate` work independently, i.e. the former
    controls only construction of a spline, and the latter only evaluation.

    When a boundary condition is 'not-a-knot' and n = 2, it is replaced by
    a condition that the first derivative is equal to the linear interpolant
    slope. When both boundary conditions are 'not-a-knot' and n = 3, the
    solution is sought as a parabola passing through given points.

    When 'not-a-knot' boundary conditions is applied to both ends, the
    resulting spline will be the same as returned by `splrep` (with ``s=0``)
    and `InterpolatedUnivariateSpline`, but these two methods use a
    representation in B-spline basis.

    .. versionadded:: 0.18.0

    Examples
    --------
    In this example the cubic spline is used to interpolate a sampled sinusoid.

    >>> from pennies.market.interpolate import CubicSplineWithNodeSens
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(10)
    >>> y = np.sin(x)
    >>> cs = CubicSplineWithNodeSens(x, y)
    >>> xs = np.arange(-0.5, 9.6, 0.1)
    >>> p = plt.figure(figsize=(6.5, 4))
    >>> p = plt.plot(x, y, 'o', label='data')
    >>> p = plt.plot(xs, np.sin(xs), label='true')
    >>> p = plt.plot(xs, cs(xs), label="S")
    >>> p = plt.xlim(-0.5, 9.5)
    >>> p = plt.legend(loc='lower left', ncol=3)
    >>> #plt.show()

    The second example is the interpolation of a polynomial y = x**3 on the
    interval 0 <= x<= 1. A cubic spline can represent this function exactly.
    To achieve that we need to specify values and first derivatives at
    endpoints of the interval. Note that y' = 3 * x**2 and thus y'(0) = 0 and
    y'(1) = 3.

    >>> cs = CubicSplineWithNodeSens([0, 0.5, 1], [0, 0.125, 1], bc_type=((1, 0), (1, 3)))
    >>> x = np.linspace(0, 1)
    >>> np.allclose(x**3, cs(x))
    True

    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    """
    def __init__(self, x, y, axis=0, bc_type='clamped', extrapolate=None):
        x, y = map(np.asarray, (x, y))

        if np.issubdtype(x.dtype, np.complexfloating):
            raise ValueError("`x` must contain real values.")

        if np.issubdtype(y.dtype, np.complexfloating):
            dtype = complex
        else:
            dtype = float
        y = y.astype(dtype, copy=False)

        axis = axis % y.ndim
        if x.ndim != 1:
            raise ValueError("`x` must be 1-dimensional.")
        if x.shape[0] < 2:
            raise ValueError("`x` must contain at least 2 elements.")
        if x.shape[0] != y.shape[axis]:
            raise ValueError("The length of `y` along `axis`={0} doesn't "
                             "match the length of `x`".format(axis))

        if not np.all(np.isfinite(x)):
            raise ValueError("`x` must contain only finite values.")
        if not np.all(np.isfinite(y)):
            raise ValueError("`y` must contain only finite values.")

        dx = np.diff(x)
        if np.any(dx <= 0):
            raise ValueError("`x` must be strictly increasing sequence.")

        n = x.shape[0]
        y = np.rollaxis(y, axis)

        bc, y = self._validate_bc(bc_type, y, y.shape[1:], axis)

        if extrapolate is None:
            if bc[0] == 'periodic':
                extrapolate = 'periodic'
            else:
                extrapolate = True

        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = np.diff(y, axis=0) / dxr

        # If bc is 'not-a-knot' this change is just a convention.
        # If bc is 'periodic' then we already checked that y[0] == y[-1],
        # and the spline is just a constant, we handle this case in the same
        # way by setting the first derivatives to slope, which is 0.
        if n == 2:
            if bc[0] in ['not-a-knot', 'periodic']:
                bc[0] = (1, slope[0])
            if bc[1] in ['not-a-knot', 'periodic']:
                bc[1] = (1, slope[0])

        # This is a very special case, when both conditions are 'not-a-knot'
        # and n == 3. In this case 'not-a-knot' can't be handled regularly
        # as both conditions are identical. We handle this case by
        # constructing a parabola passing through given points.
        if n == 3 and bc[0] == 'not-a-knot' and bc[1] == 'not-a-knot':
            A = np.zeros((3, 3))  # This is a standard matrix.
            b = np.empty((3,) + y.shape[1:], dtype=y.dtype)

            A[0, 0] = 1
            A[0, 1] = 1
            A[1, 0] = dx[1]
            A[1, 1] = 2 * (dx[0] + dx[1])
            A[1, 2] = dx[0]
            A[2, 1] = 1
            A[2, 2] = 1

            b[0] = 2 * slope[0]
            b[1] = 3 * (dxr[0] * slope[1] + dxr[1] * slope[0])
            b[2] = 2 * slope[1]

            s = solve(A, b, overwrite_a=False, overwrite_b=False,
                      check_finite=False)
        else:
            # Find derivative values at each x[i] by solving a tridiagonal
            # system.
            A = np.zeros((3, n))  # This is a banded matrix representation.
            b = np.empty((n,) + y.shape[1:], dtype=y.dtype)

            # Filling the system for i=1..n-2
            #                         (x[i-1] - x[i]) * s[i-1] +\
            # 2 * ((x[i] - x[i-1]) + (x[i+1] - x[i])) * s[i]   +\
            #                         (x[i] - x[i-1]) * s[i+1] =\
            #       3 * ((x[i+1] - x[i])*(y[i] - y[i-1])/(x[i] - x[i-1]) +\
            #           (x[i] - x[i-1])*(y[i+1] - y[i])/(x[i+1] - x[i]))

            A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
            A[0, 2:] = dx[:-1]                   # The upper diagonal
            A[-1, :-2] = dx[1:]                  # The lower diagonal

            b[1:-1] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

            bc_start, bc_end = bc

            if bc_start == 'periodic':
                # Due to the periodicity, and because y[-1] = y[0], the linear
                # system has (n-1) unknowns/equations instead of n:
                A = A[:, 0:-1]
                A[1, 0] = 2 * (dx[-1] + dx[0])
                A[0, 1] = dx[-1]

                b = b[:-1]

                # Also, due to the periodicity, the system is not tri-diagonal.
                # We need to compute a "condensed" matrix of shape (n-2, n-2).
                # See http://www.cfm.brown.edu/people/gk/chap6/node14.html for
                # more explanations.
                # The condensed matrix is obtained by removing the last column
                # and last row of the (n-1, n-1) system matrix. The removed
                # values are saved in scalar variables with the (n-1, n-1)
                # system matrix indices forming their names:
                a_m1_0 = dx[-2]  # lower left corner value: A[-1, 0]
                a_m1_m2 = dx[-1]
                a_m1_m1 = 2 * (dx[-1] + dx[-2])
                a_m2_m1 = dx[-2]
                a_0_m1 = dx[0]

                b[0] = 3 * (dxr[0] * slope[-1] + dxr[-1] * slope[0])
                b[-1] = 3 * (dxr[-1] * slope[-2] + dxr[-2] * slope[-1])

                Ac = A[:, :-1]
                b1 = b[:-1]
                b2 = np.zeros_like(b1)
                b2[0] = -a_0_m1
                b2[-1] = -a_m2_m1

                # s1 and s2 are the solutions of (n-2, n-2) system
                s1 = solve_banded((1, 1), Ac, b1, overwrite_ab=False,
                                  overwrite_b=False, check_finite=False)

                s2 = solve_banded((1, 1), Ac, b2, overwrite_ab=False,
                                  overwrite_b=False, check_finite=False)

                # computing the s[n-2] solution:
                s_m1 = ((b[-1] - a_m1_0 * s1[0] - a_m1_m2 * s1[-1]) /
                        (a_m1_m1 + a_m1_0 * s2[0] + a_m1_m2 * s2[-1]))

                # s is the solution of the (n, n) system:
                s = np.empty((n,) + y.shape[1:], dtype=y.dtype)
                s[:-2] = s1 + s_m1 * s2
                s[-2] = s_m1
                s[-1] = s[0]
            else:
                if bc_start == 'not-a-knot':
                    A[1, 0] = dx[1]
                    A[0, 1] = x[2] - x[0]
                    d = x[2] - x[0]
                    b[0] = ((dxr[0] + 2*d) * dxr[1] * slope[0] +
                            dxr[0]**2 * slope[1]) / d
                elif bc_start[0] == 1:
                    A[1, 0] = 1
                    A[0, 1] = 0
                    b[0] = bc_start[1]
                elif bc_start[0] == 2:
                    A[1, 0] = 2 * dx[0]
                    A[0, 1] = dx[0]
                    b[0] = -0.5 * bc_start[1] * dx[0]**2 + 3 * (y[1] - y[0])

                if bc_end == 'not-a-knot':
                    A[1, -1] = dx[-2]
                    A[-1, -2] = x[-1] - x[-3]
                    d = x[-1] - x[-3]
                    b[-1] = ((dxr[-1]**2*slope[-2] +
                             (2*d + dxr[-1])*dxr[-2]*slope[-1]) / d)
                elif bc_end[0] == 1:
                    A[1, -1] = 1
                    A[-1, -2] = 0
                    b[-1] = bc_end[1]
                elif bc_end[0] == 2:
                    A[1, -1] = 2 * dx[-1]
                    A[-1, -2] = dx[-1]
                    b[-1] = 0.5 * bc_end[1] * dx[-1]**2 + 3 * (y[-1] - y[-2])

                s = solve_banded((1, 1), A, b, overwrite_ab=False,
                                 overwrite_b=False, check_finite=False)

        # Compute coefficients in PPoly form.
        t = (s[:-1] + s[1:] - 2 * slope) / dxr
        c = np.empty((4, n - 1) + y.shape[1:], dtype=t.dtype)
        c[0] = t / dxr
        c[1] = (slope - s[:-1]) / dxr - t
        c[2] = s[:-1]
        c[3] = y[:-1]

        super(CubicSplineWithNodeSens, self).__init__(c, x, extrapolate=extrapolate)
        self.axis = axis

        # Compute y-derivatives at nodes
        if n < 3:
            raise NotImplementedError('node_derivatives requires more than 3 x')
        else:

            '''
            At this point A and b have been constructed with boundary conditions.
            A: stays the same.
            b:  b becomes a matrix, d(rhs_i) / dy_j

            '''

            if y.ndim > 1:  #  TODO - Confirm solution when y has more than 1 axis
                raise NotImplementedError(
                    "Solution of node_derivatives currently only allows 1D y")

            # Find vector of cross-derivative values, d/dy_j (dy(x_i)/dx)
            # Take derivative of Linear system to compute node derivatives
            # A is the same as before. New RHS is tridiagonal.
            rhs = np.zeros((n, n))  # TODO: Test for other dimensionalities
            # obtain diagonal indices for internal points
            ij_diag = tuple([np.diag_indices(n - 2)[i] + 1 for i in range(2)])
            minus_over_plus = 3 * dx[:-1] / dx[1:]
            plus_over_minus = 3 * dx[1:] / dx[:-1]
            rhs[ij_diag] = plus_over_minus - minus_over_plus  # The diagonal
            rhs[ij_diag[0], ij_diag[1] + 1] = minus_over_plus  # upper diagonal  # Confirm (+). Confirm slice
            rhs[ij_diag[0], ij_diag[1] - 1] = -plus_over_minus  # lower diagonal # Confirm (-). Confirm slice

            if bc_start[0] == 1:
                rhs[0, 0] = 0
            elif bc_start[0] == 2:
                raise NotImplementedError('bc_start not implemented. '
                                          'We only handle fixed 1st derivatives.')
                # Below is the boundary condition for dy/dx|x_0
                # b[0] = -0.5 * bc_start[1] * dx[0] ** 2 + 3 * (y[1] - y[0])
            else:
                raise NotImplementedError('bc_start not implemented. '
                                          'We only handle fixed 1st derivatives.')
            if bc_end[0] == 1:
                rhs[-1, -1] = 0
            elif bc_end[0] == 2:
                raise NotImplementedError('bc_end not implemented. '
                                          'We only handle fixed 1st derivatives.')
                # Below is the boundary condition for dy/dx|x_{n-1}
                # b[-1] = 0.5 * bc_end[1] * dx[-1] ** 2 + 3 * (y[-1] - y[-2])
            else:

                raise NotImplementedError('bc_end not implemented. '
                                          'We only handle fixed 1st derivatives.')

            d2ydydx = solve_banded((1, 1), A, rhs, overwrite_ab=False,
                             overwrite_b=False, check_finite=False)

            # The y_derivative dq(x)/dy_j
            # Create an additional vector Piecewise Polynomial
            # The PPoly weights are very similar to q(x), both fcns of x, y, y'

            inv_dx = 1 / dx
            inv_dx_rhs = inv_dx.reshape([dx.shape[0]] + [1] * (rhs.ndim - 1))
            d2_sum = (d2ydydx[:-1] + d2ydydx[1:])
            d = np.zeros((4, n - 1) + rhs.shape[1:], dtype=t.dtype)
            # Start with portion common to all j
            d[0] = (inv_dx_rhs**2) * d2_sum
            d[1] = -inv_dx_rhs * (d2_sum + d2ydydx[:-1])
            d[2] = d2ydydx[:-1]
            # Adjust when j==i: dq_i / dy_i
            ij_diag = np.diag_indices(n-1) + y.shape[1:]
            d[0][ij_diag] += 2.0 * inv_dx**3
            d[1][ij_diag] -= 3.0 * inv_dx**2
            d[3][ij_diag] += 1.0

            # Adjust when j=i+1: dq_i / dy_{i+1}
            idx_upper = (ij_diag[0], ij_diag[1] + 1)
            d[0][idx_upper] -= 2.0 * inv_dx**3
            d[1][idx_upper] += 3.0 * inv_dx**2

            self._ysens = PPoly(d, x, extrapolate=extrapolate)

    def node_derivative(self, x, nodes=None):
        """Sensitivity of y(x) to a unit move in each node's y value

        Parameters
        ----------
        x : array-like
            Points to evaluate the interpolant at.
        nodes: slice, optional
            Some way to select points to get sensitivity to

        Returns
        -------
        dy(x)/dy_i : array-like
            Shape is determined by shape of y
            the interpolation axis in the original array with the shape of x.
        """
        # TODO: Should we also allow user to select nodes???
        if nodes is not None:
            raise NotImplementedError('nodes kwarg not yet implemented.')
        else:
            return self._ysens(x)

    @staticmethod
    def _validate_bc(bc_type, y, expected_deriv_shape, axis):
        """Validate and prepare boundary conditions.

        Returns
        -------
        validated_bc : 2-tuple
            Boundary conditions for a curve start and end.
        y : ndarray
            y casted to complex dtype if one of the boundary conditions has
            complex dtype.
        """
        if isinstance(bc_type, string_types):
            if bc_type == 'periodic':
                if not np.allclose(y[0], y[-1], rtol=1e-15, atol=1e-15):
                    raise ValueError(
                        "The first and last `y` point along axis {} must "
                        "be identical (within machine precision) when "
                        "bc_type='periodic'.".format(axis))

            bc_type = (bc_type, bc_type)

        else:
            if len(bc_type) != 2:
                raise ValueError("`bc_type` must contain 2 elements to "
                                 "specify start and end conditions.")

            if 'periodic' in bc_type:
                raise ValueError("'periodic' `bc_type` is defined for both "
                                 "curve ends and cannot be used with other "
                                 "boundary conditions.")

        validated_bc = []
        for bc in bc_type:
            if isinstance(bc, string_types):
                if bc == 'clamped':
                    validated_bc.append((1, np.zeros(expected_deriv_shape)))
                elif bc == 'natural':
                    validated_bc.append((2, np.zeros(expected_deriv_shape)))
                elif bc in ['not-a-knot', 'periodic']:
                    validated_bc.append(bc)
                else:
                    raise ValueError("bc_type={} is not allowed.".format(bc))
            else:
                try:
                    deriv_order, deriv_value = bc
                except Exception:
                    raise ValueError("A specified derivative value must be "
                                     "given in the form (order, value).")

                if deriv_order not in [1, 2]:
                    raise ValueError("The specified derivative order must "
                                     "be 1 or 2.")

                deriv_value = np.asarray(deriv_value)
                if deriv_value.shape != expected_deriv_shape:
                    raise ValueError(
                        "`deriv_value` shape {} is not the expected one {}."
                        .format(deriv_value.shape, expected_deriv_shape))

                if np.issubdtype(deriv_value.dtype, np.complexfloating):
                    y = y.astype(complex, copy=False)

                validated_bc.append((deriv_order, deriv_value))

        return validated_bc, y


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def f(x):
        return np.sin(x)

    nodes_x = 0.5 * np.arange(10)  # Linearly spaced
    nodes_y = f(nodes_x)
    # nodes_y = np.stack((nodes_y, nodes_y**2), axis=1) TODO - Work on 2 lines case

    # -------------------------------------------------------------------------
    # PIECEWISE LINEAR
    # -------------------------------------------------------------------------
    pl = PiecewiseLinear(nodes_x, nodes_y, extrapolate=('clamped', 'natural'))

    x1 = -0.05
    #x1 = np.array([-0.05, 0.5])
    #x1 = np.array([[-0.05, 0.5], [4.5, 4.75]])
    y1 = pl(x1)
    f_of_x1 = f(x1)
    sens_extrap = pl.node_derivative(x1)

    sens_number = pl.node_derivative(2.25)
    sens_1d = pl.node_derivative(np.array([-0.05, 0.5]))
    sens_2d = pl.node_derivative(np.array([[-0.05, 0.5], [4.5, 4.75]]))
    xs = np.linspace(nodes_x[0] - 0.05, nodes_x[-1] + 0.5, num=100)
    pl_derivs = pl.node_derivative(xs)


    plt.figure(figsize=(6.5, 4))
    plt.title('y(x) sensitivity to a shift of x={}'.format(nodes_x[5]))
    plt.plot(xs, pl_derivs[:, 5], 'o')

    # -------------------------------------------------------------------------
    # CUBIC SPLINE
    # -------------------------------------------------------------------------
    cs_sens = CubicSplineWithNodeSens(nodes_x, nodes_y, bc_type='clamped')
    cs = CubicSpline(nodes_x, nodes_y, bc_type='clamped')
    # 'clamped' sets 1st derivative to 0.
    # 'natural' sets 2nd derivative to 0.

    print('Derivative of y(x) to shift in y values of all nodes')
    y_sens = cs_sens.node_derivative(x1)  # Evaluation node_derivatives at x1
    print('y_sens({}) = {}'.format(x1, y_sens))

    # Test results manually bumping each node
    y_shift_at_x1 = []
    y_base = cs(x1)
    for i in range(len(nodes_x)):
        y_shift = np.array(nodes_y)
        shift = 0.001
        y_shift[i] += shift
        cs_shift = CubicSpline(nodes_x, y_shift, bc_type='clamped')
        y_x1 = cs_shift(x1)
        y_shift_at_x1.append((y_x1 - y_base) / shift)
    print('bump, recalc at {}: {}'.format(x1, y_shift_at_x1))

    plt.figure(figsize=(6.5, 4))
    plt.plot(nodes_x, y_sens, 'o', label='my solution')
    plt.plot(nodes_x, y_shift_at_x1, '^', label='bump, recalc')

    plt.figure()
    plt.plot(nodes_x, y_sens - y_shift_at_x1, 'x')

    # Test that sensitivities when x is a node are 0,0,...,1,..0
    print('y_sens({}) = {}'.format(nodes_x[4], cs_sens.node_derivative(nodes_x[4])))
    print('y_sens({}) = {}'.format(nodes_x[-1], cs_sens.node_derivative(nodes_x[-1])))

    xs = np.linspace(nodes_x[0] - 0.05, nodes_x[-1] + 0.5, num=100)
    y_deriv = cs_sens.node_derivative(xs)

    plt.figure(figsize=(6.5, 4))
    plt.title('y(x) sensitivity to a shift of x={}'.format(nodes_x[5]))
    plt.plot(xs, y_deriv[:, 5], 'o')

    # Show Spline
    plt.figure(figsize=(6.5, 4))
    plt.plot(nodes_x, nodes_y, 'o', label='data')
    plt.plot(xs, f(xs), label='true')
    # our cubic vs scipy.interpolate
    plt.plot(xs, cs(xs), label="scipy")
    plt.plot(xs, cs_sens(xs), label="sens")
    # our piecewise linear vs scipy.interpolate
    plt.plot(xs, pl(xs), label='linear')
    interp_linear = interp1d(nodes_x, nodes_y, bounds_error=False, fill_value='extrapolate')
    plt.plot(xs, interp_linear(xs), label='scipy linear')

    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(loc='lower right', prop=fontP)
