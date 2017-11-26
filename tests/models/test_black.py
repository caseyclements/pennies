from __future__ import division, print_function

from pennies.models.black import price, delta, vega, theta_forward, theta_spot
from pennies.models.black import strike_from_delta, implied_vol, _d1, _d1_using_mask
import numpy as np
import pytest

def test_price():
    """Contains all tests of price()

    Test Cases for Prices
    #1. reasonable scalar values for all inputs
    #2. reasonable vector values for one input
    #3. forward = strike and sigT = 0, or sigT < 0
    #4. forward != strike and sigT = 0
    #5. forward = strike and sigT > LARGE
    #6. strike = 0
    #7. strike < 0
    #8. forward < 0
    #9. expiry < 0
    #10. forward = strike and sigT < 0
    #11. Two arrays of equal length input: forward, vol
    #12. 2 arrays without matching shapes fails 
    """


    forward, strike, expiry, vol, is_call = (100, 100, 0.5, 0.3, True)

    #1. scalar values for all inputs
    pv = price(forward, strike, expiry, vol, is_call)
    assert np.allclose(8.44700266232, pv), (
        'scalar price has changed. check expected value')

    # 2. 1d-array of inputs
    vol = np.array([-0.05, 0, 0.5, 1e15])
    prices = price(forward, strike, expiry, vol, is_call)
    
    # 10. Negative vol
    assert prices[0] == 0, (
        'Price failure when F=K and vol < 0')
    # 3. Forward==Strike and Zero vol
    assert prices[1] == 0, (
        'Price failure when F=K and vol == 0')
    assert np.allclose(prices[2],14.031620480133384, rtol=1e-10), (
        'Price failure when F=K and vol > 0')
    # 5. Large vol
    assert np.allclose(prices[3],forward, rtol=1e-10), (
        'Price failure when F=K and vol LARGE')
    
    # 12. 2 1-d arrays of different length fail 
    forward = np.array([-10.0, 0.0, 90.0, 100.0, np.inf])
    try:
        prices = price(forward, strike, expiry, vol, is_call)
        assert False, 'operands of different shapes somehow broadcast ok'
    except ValueError:
        assert True
        # But they work when new axes are added appropriately
        prices =  price(forward[:, None], strike, expiry, vol[None, :], is_call)
        assert prices.shape == (len(forward), len(vol)), (
            'Unexpected result shape')

        # 4. forward != strike, sigT == 0
        assert prices[2,1] == np.maximum(0, forward[2] - strike), (
            'Call with 0 variance remaining is not matching payoff')
        
        # 8. forward < 0
        assert np.all(np.isnan(prices[0,:])), (
            'negative forwards are producing non-nan prices')
    
    # 11.  Two arrays of equal length input: forward, vol
    assert price(forward[1:], strike, expiry, vol).shape == (4,), (
        'two equal length vectors do not result in same shaped output')

    #7. strike < 0
    assert np.isnan(price(forward=10, strike=-10, 
        maturity=2, vol=0.5, is_call=False)), (
        ' Found non-zero price for a Put struck below 0!')
    
    #9. expiry < 0
    assert np.isnan(price(forward=10, strike=9.0,
        maturity=-0.5, vol=0.5, is_call=False)), (
        ' Found non-zero price for an OTM expired option!')
    

def test_put_call_parity():
    """Tests Put-Call Parity
    
    Long a Call, and short a Put is equivalent to a Forward"""
    
    forward, strike, expiry, vol = (100, 105, 0.5, 0.3)
    call = price(forward, strike, expiry, vol, True)
    put =  price(forward, strike, expiry, vol, False)
    assert np.all(np.isclose(call - put, forward - strike)), (
        'Put-Call Parity failed')


def test_delta_large_vol_call_one():
    """Test special case of Delta when Vol is very large"""
    
    forward, strike, expiry, vol = (100, 105, 0.5, np.inf)
    assert np.isclose(delta(forward, strike, expiry, vol, True), 1.0), \
        'Delta of a Call with infinite vol should be 1'


def test_delta_large_vol_put_zero():
    """Test special case of Delta when Vol is very large"""
    
    forward, strike, expiry, vol = (100, 105, 0.5, np.inf)
    assert np.isclose(delta(forward, strike, expiry, vol, False), 0.0), \
        'Delta of a Put with infinite vol should be 0.0'


def test_strike_from_delta():
    """Test that strike of 0.5 delta is the at-the-money forward"""
    forward, delta, expiry, vol = (100, 0.5, 1.0, 0.25)
    atm_fwd = forward * np.exp(0.5 * vol ** 2 * expiry)
    assert np.isclose(strike_from_delta(forward, delta, expiry, vol), atm_fwd)


def test_implied_vol_scalars():
    """Test implied_vol returns known vol for scalar inputs"""

    forward, strike, expiry, is_call = (100, 100, 0.5, True)
    pv = 8.44700266232
    vol_expected = 0.3
    vol_obtained = implied_vol(pv, forward, strike, expiry, is_call)
    assert np.isclose(vol_expected, vol_obtained)


def test_implied_vol_vectors_1d():
    """Test implied_vol returns known vol for 1d vector inputs

    Forwards, prices, and volatility guesses share shape (3,)
    """
    f = np.array([100.0, 125.0, 150.0])
    k, t, call = (100.0, 2.0, False)
    prices = np.array([11.0, 4.0, 2.0])
    guess = np.full(prices.shape, 0.2)

    vol_expected = np.array([0.19559169, 0.20319116, 0.22963056])
    vol_obtained = implied_vol(prices, f, k, t, call, vol_guess=guess)

    assert np.all(np.isclose(vol_obtained, vol_expected))

def test_implied_vol_vectors_fail():
    """Test implied_vol raises ValueError when prices do not make sense"""
    with pytest.raises(ValueError):
        f = np.array([100.0, 125.0, 150.0])
        k, t, call = (100.0, 2.0, False)
        prices = np.full_like(f, -10.0)
        guess = np.full(prices.shape, 0.2)
        implied_vol(prices, f, k, t, call, vol_guess=guess)

    with pytest.raises(ValueError):
        prices = np.full_like(f, 0.0)
        implied_vol(prices, f, k, t, call, vol_guess=guess)

    with pytest.raises(ValueError):
        prices = np.full_like(f, np.inf)
        implied_vol(prices, f, k, t, call, vol_guess=guess)


def test_broadcasting_matching_case():
    f = np.array([100.0, 125.0, 150.0])
    k = np.array([100.0, 125.0, 125.0])
    t = np.array([0.0, 1.0, 2.0])
    vol = np.array([0.5, 0.25, 0.25])
    call = np.array([True, False, False])

    prices = price(f, k, t, vol, call)
    assert np.allclose(prices, np.array([0.0, 12.43455621, 9.26886802]))

    deltas = delta(f, k, t, vol, call)
    assert np.allclose(deltas, np.array([1.0, -0.45026178, -0.24432427]))

    strikes = strike_from_delta(f, deltas, t, vol, call)
    assert np.isnan(strikes[0])
    assert np.allclose(strikes[1:], np.array([125., 125.]))

    vegas = vega(f, k, t, vol)
    assert np.allclose(vegas, np.array([0.0, 49.47971087,  66.58770644]))

    fwd_thetas = theta_forward(f, k, t, vol, is_call=call)
    assert np.isnan(fwd_thetas[0])

    spot_theta_0 = theta_spot(f, k, t, vol,  is_call=call, interest_rate=0.0)
    assert np.isnan(spot_theta_0[0])
    assert(np.allclose(fwd_thetas[1:], spot_theta_0[1:]))

    spot_theta = theta_spot(f, k, t, vol, is_call=call, interest_rate=0.0)
    assert np.isnan(spot_theta[0])
    assert(np.allclose(spot_theta[1:], np.array([-6.18496386, -4.16173165])))

    imp_vols = implied_vol(prices, f, k, t, call, vol)
    assert np.allclose(imp_vols, vol)


def test_theta():
    forward, strike, expiry, vol, call = (100, 100, 0.5, 0.3, False)
    negative_rate = -0.1

    fwd_theta = theta_forward(forward, strike, expiry, vol, call)
    spot_theta_r0 = theta_spot(forward, strike, expiry, vol, call)
    assert np.allclose(fwd_theta, spot_theta_r0)

    spot_theta = theta_spot(forward, strike, expiry, vol,
                               is_call=call, interest_rate=negative_rate)
    assert np.allclose(spot_theta, -9.5380839937477031)


def test_d1():
    f = np.array([100.0, 125.0, 150.0])
    k = np.array([100.0, 125.0, 125.0])
    t = np.array([0.0, 1.0, 2.0])
    vol = np.array([0.5, 0.25, 0.25])
    assert np.allclose(_d1(f, k, vol * np.sqrt(t)),
                       _d1_using_mask(f, k, vol * np.sqrt(t)))