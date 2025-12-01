"""
Options Pricing & Risk Analytics Engine - Python Implementation
Author: Soham Gugale
Description: Black-Scholes pricing, Monte Carlo simulation, Greeks calculation, and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BlackScholesAnalytics:
    """Black-Scholes option pricing and Greeks calculation"""
    
    def __init__(self, S, K, T, r, sigma):
        """
        Initialize Black-Scholes model
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (years)
        r (float): Risk-free rate
        sigma (float): Volatility
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def d1(self):
        """Calculate d1 parameter"""
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        """Calculate d2 parameter"""
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_price(self):
        """Calculate call option price"""
        return self.S * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())
    
    def put_price(self):
        """Calculate put option price"""
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S * norm.cdf(-self.d1())
    
    def delta_call(self):
        """Calculate delta for call option"""
        return norm.cdf(self.d1())
    
    def delta_put(self):
        """Calculate delta for put option"""
        return -norm.cdf(-self.d1())
    
    def gamma(self):
        """Calculate gamma (same for call and put)"""
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        """Calculate vega (same for call and put)"""
        return self.S * norm.pdf(self.d1()) * np.sqrt(self.T) / 100
    
    def theta_call(self):
        """Calculate theta for call option (per day)"""
        term1 = -self.S * norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T))
        term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())
        return (term1 + term2) / 365
    
    def theta_put(self):
        """Calculate theta for put option (per day)"""
        term1 = -self.S * norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T))
        term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2())
        return (term1 + term2) / 365


def monte_carlo_pricing(S, K, T, r, sigma, num_sims=500000):
    """
    Monte Carlo simulation for option pricing
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity
    r (float): Risk-free rate
    sigma (float): Volatility
    num_sims (int): Number of simulations
    
    Returns:
    float: Call option price
    """
    np.random.seed(42)
    Z = np.random.standard_normal(num_sims)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    call_payoffs = np.maximum(ST - K, 0)
    call_price = np.exp(-r * T) * np.mean(call_payoffs)
    return call_price


def plot_option_payoff():
    """Plot option payoff diagrams"""
    S_range = np.linspace(50, 150, 100)
    K = 100
    
    call_payoffs = np.maximum(S_range - K, 0)
    put_payoffs = np.maximum(K - S_range, 0)
    
    plt.figure(figsize=(12, 5))
    
    # Call option payoff
    plt.subplot(1, 2, 1)
    plt.plot(S_range, call_payoffs, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=K, color='r', linestyle='--', alpha=0.3, label=f'Strike = ${K}')
    plt.xlabel('Stock Price at Expiration ($)', fontsize=11)
    plt.ylabel('Payoff ($)', fontsize=11)
    plt.title('Call Option Payoff Diagram', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Put option payoff
    plt.subplot(1, 2, 2)
    plt.plot(S_range, put_payoffs, 'r-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=K, color='r', linestyle='--', alpha=0.3, label=f'Strike = ${K}')
    plt.xlabel('Stock Price at Expiration ($)', fontsize=11)
    plt.ylabel('Payoff ($)', fontsize=11)
    plt.title('Put Option Payoff Diagram', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/option_payoffs.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/option_payoffs.png")
    plt.close()


def plot_option_prices():
    """Plot option prices vs stock price"""
    S_range = np.linspace(50, 150, 100)
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    call_prices = []
    put_prices = []
    
    for S in S_range:
        bs = BlackScholesAnalytics(S, K, T, r, sigma)
        call_prices.append(bs.call_price())
        put_prices.append(bs.put_price())
    
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, call_prices, 'b-', linewidth=2.5, label='Call Price')
    plt.plot(S_range, put_prices, 'r-', linewidth=2.5, label='Put Price')
    plt.axvline(x=K, color='gray', linestyle='--', alpha=0.5, label='Strike = $100')
    plt.xlabel('Stock Price ($)', fontsize=12)
    plt.ylabel('Option Price ($)', fontsize=12)
    plt.title('Black-Scholes Option Prices\n(K=$100, T=1yr, σ=20%, r=5%)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/option_prices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/option_prices.png")
    plt.close()


def plot_greeks():
    """Plot Greeks vs stock price"""
    S_range = np.linspace(50, 150, 100)
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    deltas, gammas, vegas, thetas = [], [], [], []
    
    for S in S_range:
        bs = BlackScholesAnalytics(S, K, T, r, sigma)
        deltas.append(bs.delta_call())
        gammas.append(bs.gamma())
        vegas.append(bs.vega())
        thetas.append(bs.theta_call())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Delta
    axes[0, 0].plot(S_range, deltas, 'b-', linewidth=2.5)
    axes[0, 0].axvline(x=K, color='r', linestyle='--', alpha=0.3, label='Strike')
    axes[0, 0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    axes[0, 0].set_xlabel('Stock Price ($)', fontsize=11)
    axes[0, 0].set_ylabel('Delta', fontsize=11)
    axes[0, 0].set_title('Delta: Option Price Sensitivity to Stock Price', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    
    # Gamma
    axes[0, 1].plot(S_range, gammas, 'g-', linewidth=2.5)
    axes[0, 1].axvline(x=K, color='r', linestyle='--', alpha=0.3, label='Strike')
    axes[0, 1].set_xlabel('Stock Price ($)', fontsize=11)
    axes[0, 1].set_ylabel('Gamma', fontsize=11)
    axes[0, 1].set_title('Gamma: Rate of Change of Delta', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)
    
    # Vega
    axes[1, 0].plot(S_range, vegas, 'm-', linewidth=2.5)
    axes[1, 0].axvline(x=K, color='r', linestyle='--', alpha=0.3, label='Strike')
    axes[1, 0].set_xlabel('Stock Price ($)', fontsize=11)
    axes[1, 0].set_ylabel('Vega', fontsize=11)
    axes[1, 0].set_title('Vega: Sensitivity to Volatility', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)
    
    # Theta
    axes[1, 1].plot(S_range, thetas, 'c-', linewidth=2.5)
    axes[1, 1].axvline(x=K, color='r', linestyle='--', alpha=0.3, label='Strike')
    axes[1, 1].axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    axes[1, 1].set_xlabel('Stock Price ($)', fontsize=11)
    axes[1, 1].set_ylabel('Theta ($/day)', fontsize=11)
    axes[1, 1].set_title('Theta: Time Decay', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/greeks_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/greeks_analysis.png")
    plt.close()


def plot_volatility_surface():
    """Plot volatility surface (strikes vs maturities)"""
    strikes = np.linspace(80, 120, 20)
    maturities = np.linspace(0.1, 2.0, 20)
    S = 100
    r = 0.05
    sigma = 0.2
    
    K_grid, T_grid = np.meshgrid(strikes, maturities)
    call_prices = np.zeros_like(K_grid)
    
    for i in range(len(maturities)):
        for j in range(len(strikes)):
            bs = BlackScholesAnalytics(S, strikes[j], maturities[i], r, sigma)
            call_prices[i, j] = bs.call_price()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(K_grid, T_grid, call_prices, cmap='viridis', alpha=0.9, edgecolor='none')
    ax.set_xlabel('Strike Price ($)', fontsize=11, labelpad=10)
    ax.set_ylabel('Time to Maturity (years)', fontsize=11, labelpad=10)
    ax.set_zlabel('Call Price ($)', fontsize=11, labelpad=10)
    ax.set_title('Option Price Surface\n(S=$100, σ=20%, r=5%)', fontsize=13, fontweight='bold', pad=20)
    ax.view_init(elev=25, azim=45)
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    plt.savefig('outputs/volatility_surface.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/volatility_surface.png")
    plt.close()


def validate_with_market_data():
    """Validate model with real market data"""
    print("\n" + "="*60)
    print("MARKET DATA VALIDATION")
    print("="*60)
    
    # Example with Apple stock (you can change to any ticker)
    ticker = "AAPL"
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get current stock price
        current_price = stock.info.get('currentPrice', None)
        if current_price is None:
            current_price = stock.info.get('regularMarketPrice', 100.0)
        
        print(f"\n{ticker} Current Price: ${current_price:.2f}")
        
        # Get options data
        exp_dates = stock.options
        if len(exp_dates) > 0:
            options = stock.option_chain(exp_dates[0])
            calls = options.calls.head(10)
            
            print(f"\nOptions expiring: {exp_dates[0]}")
            print("\nSample Call Options (First 5):")
            print("-" * 60)
            display_cols = ['strike', 'lastPrice', 'impliedVolatility']
            print(calls[display_cols].head(5).to_string(index=False))
            
            # Calculate days to expiration
            exp_date = datetime.strptime(exp_dates[0], '%Y-%m-%d')
            days_to_exp = (exp_date - datetime.now()).days
            T = max(days_to_exp / 365.0, 0.01)  # Avoid division by zero
            
            # Use risk-free rate (approximate)
            r = 0.05
            
            print(f"\nDays to Expiration: {days_to_exp}")
            print(f"Time to Maturity (years): {T:.4f}")
            
            # Compare model vs market for first ATM call
            atm_idx = (calls['strike'] - current_price).abs().idxmin()
            atm_call = calls.loc[atm_idx]
            
            K = atm_call['strike']
            market_price = atm_call['lastPrice']
            implied_vol = atm_call['impliedVolatility']
            
            if implied_vol > 0 and market_price > 0:
                bs = BlackScholesAnalytics(current_price, K, T, r, implied_vol)
                model_price = bs.call_price()
                
                error = abs(model_price - market_price) / market_price * 100
                
                print(f"\n{'─'*60}")
                print("ATM CALL OPTION COMPARISON")
                print(f"{'─'*60}")
                print(f"Strike Price:    ${K:.2f}")
                print(f"Market Price:    ${market_price:.2f}")
                print(f"Model Price:     ${model_price:.2f}")
                print(f"Pricing Error:   {error:.2f}%")
                print(f"Implied Vol:     {implied_vol*100:.2f}%")
                print(f"{'─'*60}")
            else:
                print("\nInsufficient option data for validation")
        else:
            print("\nNo options data available for this ticker")
            
    except Exception as e:
        print(f"\nCould not fetch market data: {e}")
        print("Using synthetic data for demonstration...")
        print("\nNote: Market data validation requires internet connection")
        print("and valid ticker symbol. The model still works with")
        print("theoretical parameters as shown in other sections.")


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("OPTIONS PRICING & RISK ANALYTICS ENGINE")
    print("="*60)
    print("Python Implementation - Black-Scholes & Monte Carlo\n")
    
    # Example parameters
    S = 100.0   # Stock price
    K = 100.0   # Strike price
    T = 1.0     # 1 year
    r = 0.05    # 5% risk-free rate
    sigma = 0.2 # 20% volatility
    
    # Black-Scholes pricing
    print("─" * 60)
    print("BLACK-SCHOLES ANALYTICAL PRICING")
    print("─" * 60)
    bs = BlackScholesAnalytics(S, K, T, r, sigma)
    print(f"Stock Price (S):     ${S:.2f}")
    print(f"Strike Price (K):    ${K:.2f}")
    print(f"Time to Maturity:    {T:.2f} years")
    print(f"Risk-free Rate:      {r*100:.2f}%")
    print(f"Volatility (σ):      {sigma*100:.2f}%")
    print(f"\nCall Option Price:   ${bs.call_price():.4f}")
    print(f"Put Option Price:    ${bs.put_price():.4f}")
    
    # Greeks
    print("\n" + "─" * 60)
    print("GREEKS (Risk Measures)")
    print("─" * 60)
    print(f"Delta (Call):        {bs.delta_call():.4f}")
    print(f"Gamma:               {bs.gamma():.4f}")
    print(f"Vega:                {bs.vega():.4f}")
    print(f"Theta (Call):        ${bs.theta_call():.4f}/day")
    
    # Monte Carlo
    print("\n" + "─" * 60)
    print("MONTE CARLO SIMULATION")
    print("─" * 60)
    num_sims = 500000
    print(f"Number of simulations: {num_sims:,}")
    mc_price = monte_carlo_pricing(S, K, T, r, sigma, num_sims=num_sims)
    print(f"Call Option Price:     ${mc_price:.4f}")
    
    error = abs(bs.call_price() - mc_price) / bs.call_price() * 100
    print(f"\nValidation:")
    print(f"Black-Scholes Price:   ${bs.call_price():.4f}")
    print(f"Monte Carlo Price:     ${mc_price:.4f}")
    print(f"Pricing Error:         {error:.4f}%")
    
    # Generate visualizations
    print("\n" + "─" * 60)
    print("GENERATING VISUALIZATIONS")
    print("─" * 60)
    
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created outputs/ directory")
    
    plot_option_payoff()
    plot_option_prices()
    plot_greeks()
    plot_volatility_surface()
    
    # Market validation
    validate_with_market_data()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nAll visualization files saved to outputs/ directory:")
    print("  • option_payoffs.png")
    print("  • option_prices.png")
    print("  • greeks_analysis.png")
    print("  • volatility_surface.png")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()