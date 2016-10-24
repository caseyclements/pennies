import pytest
import numpy as np
from scipy.interpolate import CubicSpline
from pennies.market.interpolate import CubicSplineWithNodeSens, PiecewiseLinear


def f(z):
    return np.sin(z)

n = 10
x = 0.5 * np.arange(n)  # Linearly spaced
y = f(x)


def test_cubic_unit_sens_when_x_is_a_node():
    """Sanity check of unit sensitivity"""

    cs_sens = CubicSplineWithNodeSens(x, y, bc_type='clamped')
    ix = np.arange(len(x))

    sens_inner_node = cs_sens.node_derivative(x[4])
    assert np.isclose(sens_inner_node[4], 1.0)
    assert np.allclose(sens_inner_node[ix != 4], 0.0)
    assert np.isclose(1.0, np.sum(sens_inner_node))

    sens_first_node = cs_sens.node_derivative(x[0])
    assert np.isclose(sens_first_node[0], 1.0)
    assert np.allclose(sens_first_node[ix != 0], 0.0)
    assert np.isclose(1.0, np.sum(sens_first_node))

    sens_last_node = cs_sens.node_derivative(x[-1])
    assert np.isclose(sens_last_node[-1], 1.0)
    assert np.allclose(sens_last_node[ix != (n-1)], 0.0)
    assert np.isclose(1.0, np.sum(sens_last_node))


def test_cubic_sens_against_manual_bumping():
    """Test results by manually bumping each node."""
    cs_sens = CubicSplineWithNodeSens(x, y, bc_type='clamped')
    x1 = 2.25
    y_sens = cs_sens.node_derivative(x1)  # Evaluation node_derivatives at x1
    cs = CubicSpline(x, y, bc_type='clamped')

    y_shift_at_x1 = []
    y_base = cs(x1)
    for i in range(len(x)):
        y_shift = np.array(y)
        shift = 0.001
        y_shift[i] += shift
        cs_shift = CubicSpline(x, y_shift, bc_type='clamped')
        y_x1 = cs_shift(x1)
        y_shift_at_x1.append((y_x1 - y_base) / shift)

    assert np.allclose(y_sens, y_shift_at_x1)

def test_piecewise_linear_unit_sens_when_x_is_a_node():
    """Sanity check of unit sensitivity"""

    pl_sens = PiecewiseLinear(x, y, extrapolate=('clamped', 'natural'))
    ix = np.arange(len(x))

    sens_inner_node = pl_sens.node_derivative(x[4])
    assert np.isclose(sens_inner_node[4], 1.0)
    assert np.allclose(sens_inner_node[ix != 4], 0.0)
    assert np.isclose(1.0, np.sum(sens_inner_node))

    sens_first_node = pl_sens.node_derivative(x[0])
    assert np.isclose(sens_first_node[0], 1.0)
    assert np.allclose(sens_first_node[ix != 0], 0.0)
    assert np.isclose(1.0, np.sum(sens_first_node))

    sens_last_node = pl_sens.node_derivative(x[-1])
    assert np.isclose(sens_last_node[-1], 1.0)
    assert np.allclose(sens_last_node[ix != (n-1)], 0.0)
    assert np.isclose(1.0, np.sum(sens_last_node))


def test_piecewise_linear_sens_locality():
    """Test that y(x) is sensitive only to its nearest nodes,
    the one to its left and the one to its right."""
    pl_sens = PiecewiseLinear(x, y, extrapolate=('clamped', 'natural'))
    sens_inner = pl_sens.node_derivative(0.5 * (x[4] + x[5]))
    assert(np.isclose(sens_inner[4], 0.5))
    assert (np.isclose(sens_inner[5], 0.5))
    assert (np.isclose(np.sum(sens_inner[:4]), 0.0))
    assert (np.isclose(np.sum(sens_inner[5+1:]), 0.0))


def test_piecewise_linear_sens_scalar_x():
    """Test shape of output when asked for derivative of scalar."""
    pl_sens = PiecewiseLinear(x, y, extrapolate=('clamped', 'natural'))
    eval_at = 0.5 * (x[4] + x[5])
    sens = pl_sens.node_derivative(eval_at)
    assert(sens.ndim == 1)
    assert(sens.shape == (len(x),))


def test_piecewise_linear_sens_1d_x():
    """Test shape of output when asked for derivative of 1D array."""
    pl_sens = PiecewiseLinear(x, y, extrapolate=('clamped', 'natural'))
    eval_at = x + 0.5
    eval_shape = eval_at.shape
    sens = pl_sens.node_derivative(eval_at)
    assert(sens.ndim == 2)
    assert(sens.shape == (eval_shape + (len(x),)))


def test_piecewise_linear_sens_2d_x():
    """Test shape of output when asked for derivative of 2D array."""
    pl_sens = PiecewiseLinear(x, y, extrapolate=('clamped', 'natural'))
    eval_shape = (3, 5)
    eval_at = np.random.random(eval_shape) * x.max()
    sens = pl_sens.node_derivative(eval_at)
    assert(sens.ndim == 3)
    assert(sens.shape == (eval_shape + (len(x),)))


def test_piecewise_linear_sens_3d_x():
    """Test shape of output when asked for derivative of 3D array."""
    pl_sens = PiecewiseLinear(x, y, extrapolate=('clamped', 'natural'))
    eval_shape = (2, 3, 4)
    eval_at = np.random.random(eval_shape) * x.max()
    sens = pl_sens.node_derivative(eval_at)
    assert(sens.ndim == 4)
    assert(sens.shape == (eval_shape + (len(x),)))


def test_piecewise_linear_sens_extrap():
    """Test extrapolation of PiecewiseLinear."""
    pl_sens = PiecewiseLinear(x, y, extrapolate=('clamped', 'natural'))
    eval_at = np.array([x[0] - 100.0, x[-1] + 2.0])
    sens = pl_sens.node_derivative(eval_at)
    assert(sens.ndim == 2)
    assert(sens.shape == (2, len(x)))

    # Test left extrapolation.
    # As it is 'clamped', all should be 0, except for 0'th node, == 1
    assert(np.isclose(np.sum(sens[0][1:]), 0.0))
    assert (np.isclose(sens[0][0], 1.0))

    # Test right extrapolation
    # As it is 'natural', it will be sensitive to the 2 nearest nodes
    z = x[-1] + 2
    sens_m1 = 1 + (z - x[-1]) / (x[-1] - x[-2])
    sens_m2 = - (z - x[-1]) / (x[-1] - x[-2])
    assert (np.isclose(sens[1][-1], sens_m1))
    assert (np.isclose(sens[1][-2], sens_m2))


def test_order_of_outputs():
    """ Confirm output shape is consistent for PiecewiseLinear and Spline."""
    pl = PiecewiseLinear(x, y, extrapolate=('clamped', 'natural'))
    cs = CubicSplineWithNodeSens(x, y, bc_type='clamped')

    eval_shape = (3, 5)
    eval_at = np.random.random(eval_shape) * x.max()

    pl_sens = pl.node_derivative(eval_at)
    cs_sens = cs.node_derivative(eval_at)
    assert(pl_sens.shape == cs_sens.shape)


def test_cubic_docstring_1d():
    """Reproduces example 1 run in docstring"""
    from pennies.market.interpolate import CubicSplineWithNodeSens
    import matplotlib.pyplot as plt
    x = np.arange(10)
    y = np.sin(x)
    cs = CubicSplineWithNodeSens(x, y)
    xs = np.arange(-0.5, 9.6, 0.1)

    p = plt.figure(figsize=(6.5, 4))
    p = plt.plot(x, y, 'o', label='data')
    p = plt.plot(xs, np.sin(xs), label='true')
    p = plt.plot(xs, cs(xs), label="S")
    p = plt.xlim(-0.5, 9.5)
    p = plt.legend(loc='lower left', ncol=3)
    #plt.show()


def test_cubic_docstring_1d_cubic_function_exact_match():
    """Test that we exactly reproduce a cubic function"""
    from pennies.market.interpolate import CubicSplineWithNodeSens
    cs = CubicSplineWithNodeSens([0, 0.5, 1], [0, 0.125, 1], bc_type=((1, 0), (1, 3)))
    x = np.linspace(0, 1)
    assert np.allclose(x ** 3, cs(x))


def test_cubic_docstring_2d_unitcircle():
    """Test of 2D dependent variable, y"""

    '''
    ---------------------------------------------------------------
    NOTE: The following came from the CubicSpline documentation in scipy.
    When 2d is working, put this, or something similar back into docstring
    for CubicSplineWithNodeSens.
    ---------------------------------------------------------------
    In the third example, the unit circle is interpolated with a spline. A
    periodic boundary condition is used. You can see that the first derivative
    values, ds/dx=0, ds/dy=1 at the periodic point (1, 0) are correctly
    computed. Note that a circle cannot be exactly represented by a cubic
    spline. To increase precision, more breakpoints would be required.

    >>> theta = 2 * np.pi * np.linspace(0, 1, 5)
    >>> y = np.c_[np.cos(theta), np.sin(theta)]
    >>> cs = CubicSplineWithNodeSens(theta, y, bc_type='periodic')
    >>> print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
    ds/dx=0.0 ds/dy=1.0
    >>> xs = 2 * np.pi * np.linspace(0, 1, 100)
    >>> p = plt.figure(figsize=(6.5, 4))
    >>> p = plt.plot(y[:, 0], y[:, 1], 'o', label='data')
    >>> p = plt.plot(np.cos(xs), np.sin(xs), label='true')
    >>> p = plt.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
    >>> p = plt.axes().set_aspect('equal')
    >>> p = plt.legend(loc='center')
    >>> #plt.show()
    '''
    with pytest.raises(NotImplementedError):  # TODO 2D
        from pennies.market.interpolate import CubicSplineWithNodeSens
        import matplotlib.pyplot as plt
        theta = 2 * np.pi * np.linspace(0, 1, 5)
        y = np.c_[np.cos(theta), np.sin(theta)]
        cs = CubicSplineWithNodeSens(theta, y, bc_type='periodic')
        print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
        #ds/dx=0.0 ds/dy=1.0
        xs = 2 * np.pi * np.linspace(0, 1, 100)
        p = plt.figure(figsize=(6.5, 4))
        p = plt.plot(y[:, 0], y[:, 1], 'o', label='data')
        p = plt.plot(np.cos(xs), np.sin(xs), label='true')
        p = plt.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
        p = plt.axes().set_aspect('equal')
        p = plt.legend(loc='center')
        #plt.show()



if __name__ == '__main__':
    #test_cubic_sens_against_manual_bumping()
    #test_cubic_unit_sens_when_x_is_a_node()
    #test_piecewise_linear_unit_sens_when_x_is_a_node()
    #test_piecewise_linear_sens_locality()
    #test_piecewise_linear_sens_scalar_x()
    #test_piecewise_linear_sens_1d_x()
    #test_piecewise_linear_sens_2d_x()
    #test_piecewise_linear_sens_3d_x()
    #test_piecewise_linear_sens_extrap()
    #test_order_of_outputs()
    #test_cubic_docstring_1d()
    test_cubic_docstring_2d_unitcircle()
    #test_cubic_docstring_1d_cubic_function_exact_match()
