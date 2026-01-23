import unittest
import numpy as np
from scipy.stats import norm
from advanced_options_engine import FDMEngine, BlackScholesEngine

class TestOptionPricing(unittest.TestCase):
    
    def test_european_call_convergence(self):
        """Verify FDM matches Black-Scholes for European Call"""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        # Analytical
        bs = BlackScholesEngine(S, K, T, r, sigma, is_call=True)
        bs_price = bs.price()
        
        # Numerical
        fdm = FDMEngine(S, K, T, r, sigma, is_call=True, is_american=False)
        fdm_res = fdm.calculate(price_steps=200, time_steps=2000)
        
        print(f"BS: {bs_price:.4f}, FDM: {fdm_res['price']:.4f}")
        self.assertAlmostEqual(bs_price, fdm_res['price'], delta=0.05)

    def test_put_call_parity(self):
        """Verify Call - Put = S - K*exp(-rT)"""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        call = FDMEngine(S, K, T, r, sigma, is_call=True, is_american=False).calculate()['price']
        put = FDMEngine(S, K, T, r, sigma, is_call=False, is_american=False).calculate()['price']
        
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        
        self.assertAlmostEqual(lhs, rhs, delta=0.05)

    def test_american_premium(self):
        """American Put should be >= European Put"""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        euro_put = FDMEngine(S, K, T, r, sigma, is_call=False, is_american=False).calculate()['price']
        amer_put = FDMEngine(S, K, T, r, sigma, is_call=False, is_american=True).calculate()['price']
        
        self.assertTrue(amer_put >= euro_put)

if __name__ == '__main__':
    unittest.main()
