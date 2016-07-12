from __future__ import division, print_function

from ..black import price, delta, strike_from_delta
import numpy as np

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


    forward, strike, expiry, vol, isCall = (100, 100, 0.5, 0.3, True)

    #1. scalar values for all inputs
    pv = price(forward, strike, expiry, vol, isCall)
    assert np.allclose(8.44700266232, pv), (
        'scalar price has changed. check expected value')

    # 2. 1d-array of inputs
    vol = np.array([-0.05, 0, 0.5, 1e15])
    prices = price(forward, strike, expiry, vol, isCall)
    
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
        prices = price(forward, strike, expiry, vol, isCall)
        assert False, 'operands of different shapes somehow broadcast ok'
    except ValueError:
        assert True
        # But they work when new axes are added appropriately
        prices =  price(forward[:, None], strike, expiry, vol[None, :], isCall)
        assert prices.shape == (len(forward), len(vol)), (
            'Unexpected result shape')

        # 4. forward != strike, sigT == 0
        assert prices[2,1] == np.maximum(0, forward[2] - strike), (
            'Call with 0 variance remaining is not matching payoff')
        
        # 8. forward < 0
        print('negative forward prices = {}'.format(prices[0,:]))
        assert np.all(np.isnan(prices[0,:])), (
            'negative forwards are producing non-nan prices')
    
    # 11.  Two arrays of equal length input: forward, vol
    assert price(forward[1:], strike, expiry, vol).shape == (4,), (
        'two equal length vectors do not result in same shaped output')

    #7. strike < 0
    assert np.isnan(price(forward=10, strike=-10, 
        maturity=2, vol=0.5, isCall=False)), (
        ' Found non-zero price for a Put struck below 0!')
    
    #9. expiry < 0
    assert np.isnan(price(forward=10, strike=9.0,
        maturity=-0.5, vol=0.5, isCall=False)), (
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
    assert np.isclose(delta(forward, strike, expiry, vol, True), 1.0), (
        'Delta of a Call with infinite vol should be 1')    


def test_delta_large_vol_put_zero():
    """Test special case of Delta when Vol is very large"""
    
    forward, strike, expiry, vol = (100, 105, 0.5, np.inf)
    assert np.isclose(delta(forward, strike, expiry, vol, False), 0.0), (
        'Delta of a Put with infinite vol should be 0.0')


def test_strike_from_delta():
    """Test that strike of 0.5 delta is the at-the-money forward"""
    forward, delta, expiry, vol = (100, 0.5, 1.0, 0.25)
    atm_forward = forward * np.exp(0.5 * vol ** 2 * expiry)
    assert np.isclose(strike_from_delta(forward, delta, expiry, vol), atm_forward)

