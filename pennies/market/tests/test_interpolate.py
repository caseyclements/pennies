import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
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


if __name__ == '__main__':
    #test_cubic_sens_against_manual_bumping()
    #test_cubic_unit_sens_when_x_is_a_node()
    #test_piecewise_linear_unit_sens_when_x_is_a_node()
    #test_piecewise_linear_sens_locality()
    test_piecewise_linear_sens_scalar_x()
    #test_piecewise_linear_sens_1d_x()
    #test_piecewise_linear_sens_2d_x()
    #test_piecewise_linear_sens_3d_x()
    #test_piecewise_linear_sens_extrap()
    #test_order_of_outputs()
