import numpy as np
from scipy.stats import norm

# --- C++ INTEGRATION ---
try:
    import options_solver
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

# --- PYTHON FALLBACK (Analytical Black-Scholes) ---
class BlackScholesEngine:
    def __init__(self, S, K, T, r, sigma, is_call=True):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.is_call = is_call

    def price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.is_call:
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

# --- C++ WRAPPER (Numerical PDE Solver) ---
class AmericanOptionPricer:
    def __init__(self, S, K, T, r, sigma, is_call=True):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.is_call = is_call

    def price(self, price_steps=100, time_steps=1000):
        if not CPP_AVAILABLE:
            return None
        return options_solver.price_american(
            self.S, self.K, self.T, self.r, self.sigma,
            price_steps, time_steps, self.is_call
        )
