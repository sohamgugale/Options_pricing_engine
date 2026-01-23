import numpy as np
from scipy.stats import norm

# --- C++ INTEGRATION ---
try:
    import options_solver
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

# --- C++ WRAPPER (Now handles Greeks & Barriers) ---
class FDMEngine:
    def __init__(self, S, K, T, r, sigma, is_call=True, is_american=True, barrier=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.is_call = is_call
        self.is_american = is_american
        self.barrier = barrier

    def calculate(self, price_steps=200, time_steps=2000):
        if not CPP_AVAILABLE:
            return None
            
        # Call C++ Solver
        results = options_solver.solve(
            self.S, self.K, self.T, self.r, self.sigma,
            self.barrier,
            price_steps, time_steps, 
            self.is_call, self.is_american
        )
        return results

# --- ANALYTICAL FALLBACK (For Comparison) ---
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
