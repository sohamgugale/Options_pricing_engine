"""Accuracy and timing checks for the C++ finite-difference engine.

Run after `pip install .`:  python validate.py
"""
import time

import numpy as np
from scipy.stats import norm

import options_solver as fdm

K, R = 100.0, 0.05


def bs_price(S, K, T, r, sigma, is_call):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def binomial_american_put(S, K, T, r, sigma, steps=5000):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    V = np.maximum(K - S * u ** (steps - j) * d ** j, 0.0)
    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        ST = S * u ** (i - j) * d ** j
        V = np.maximum(disc * (p * V[:-1] + (1 - p) * V[1:]), K - ST)
    return V[0]


def main():
    # 1. European accuracy vs closed form
    errors = []
    for S in [80, 90, 100, 110, 120]:
        for T in [0.25, 0.5, 1.0]:
            for sigma in [0.15, 0.25, 0.40]:
                for is_call in [True, False]:
                    num = fdm.solve(S, K, T, R, sigma, 0.0, 400, 400, is_call, False)["price"]
                    ref = bs_price(S, K, T, R, sigma, is_call)
                    # relative error, or absolute error for near-worthless options
                    errors.append(abs(num - ref) / ref if ref > 0.5 else abs(num - ref))
    print(f"European, {len(errors)} cases: max error {max(errors):.2%}, median {np.median(errors):.2%}")

    # 2. American put vs binomial tree
    amer = fdm.solve(100, K, 1.0, R, 0.2, 0.0, 400, 400, False, True)["price"]
    tree = binomial_american_put(100, K, 1.0, R, 0.2)
    print(f"American put: FDM {amer:.4f} vs binomial {tree:.4f} ({abs(amer - tree) / tree:.2%})")

    # 3. Runtime: price + all Greeks
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        fdm.solve(100, K, 1.0, R, 0.2, 0.0, 200, 200, False, True)
    print(f"Price + Greeks, 200x200 grid: {(time.perf_counter() - t0) / n * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
