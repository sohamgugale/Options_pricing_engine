"""
Standalone Delta Hedging Backtest Demo
No external dependencies except numpy (built-in)
"""

import numpy as np
from datetime import datetime, timedelta

# Mini Black-Scholes implementation
def norm_cdf(x):
    """Standard normal CDF"""
    return 0.5 * (1.0 + np.tanh(x / np.sqrt(2)))

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)

def call_delta(S, K, T, r, sigma):
    """Call option delta"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm_cdf(d1)

def call_gamma(S, K, T, r, sigma):
    """Call option gamma"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    pdf = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    return pdf / (S * sigma * np.sqrt(T))


class DeltaHedgingBacktest:
    """Complete delta hedging backtest"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.results = []
        
    def run(self, S0=100, K=100, T=30/252, r=0.05, sigma=0.20,
            contracts=100, rehedge_freq=5, num_days=30):
        """
        Run delta hedging backtest
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            contracts: Number of option contracts to sell
            rehedge_freq: Rehedge every N days
            num_days: Total days to simulate
        """
        
        print("="*70)
        print("DELTA HEDGING STRATEGY BACKTEST")
        print("="*70)
        print(f"\nStrategy: Sell {contracts} ATM call contracts, delta hedge")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Stock Price: ${S0:.2f}")
        print(f"Strike: ${K:.2f}")
        print(f"Time to Maturity: {T*252:.0f} days")
        print(f"Volatility: {sigma*100:.1f}%")
        print(f"Rehedge Frequency: Every {rehedge_freq} days\n")
        
        # Generate stock price path (Geometric Brownian Motion)
        np.random.seed(42)
        dt = 1/252
        stock_prices = [S0]
        
        for _ in range(num_days):
            dW = np.random.normal(0, 1)
            S_new = stock_prices[-1] * np.exp((r - 0.5*sigma**2)*dt + 
                                             sigma*np.sqrt(dt)*dW)
            stock_prices.append(S_new)
        
        # Initialize positions
        cash = self.initial_capital
        stock_position = 0
        option_position = -contracts  # Sold contracts
        
        # Day 0: Sell options and initial hedge
        T_current = T
        option_price = black_scholes_call(stock_prices[0], K, T_current, r, sigma)
        delta = call_delta(stock_prices[0], K, T_current, r, sigma)
        gamma = call_gamma(stock_prices[0], K, T_current, r, sigma)
        
        # Receive premium
        premium_received = option_price * contracts * 100
        cash += premium_received
        
        # Initial hedge
        shares_needed = -delta * contracts * 100
        stock_position = shares_needed
        cash -= shares_needed * stock_prices[0]
        
        print(f"Day 0: Initial Setup")
        print(f"  Option Price: ${option_price:.2f}")
        print(f"  Premium Received: ${premium_received:,.2f}")
        print(f"  Delta: {delta:.4f}")
        print(f"  Initial Hedge: Buy {shares_needed:,.0f} shares @ ${stock_prices[0]:.2f}")
        print(f"  Cash after hedge: ${cash:,.2f}\n")
        
        # Track results
        for day in range(1, num_days + 1):
            S = stock_prices[day]
            T_current = max(T - day/252, 0.001)
            
            # Current option value
            option_value = black_scholes_call(S, K, T_current, r, sigma)
            delta = call_delta(S, K, T_current, r, sigma)
            gamma = call_gamma(S, K, T_current, r, sigma)
            
            # Portfolio value
            option_mtm = option_value * contracts * 100
            stock_mtm = stock_position * S
            portfolio_value = cash + stock_mtm - option_mtm
            pnl = portfolio_value - self.initial_capital
            
            # Rehedge if needed
            rehedged = False
            if day % rehedge_freq == 0:
                target_shares = -delta * contracts * 100
                shares_to_trade = target_shares - stock_position
                
                # Execute trade (with 0.1% transaction cost)
                trade_cost = abs(shares_to_trade * S) * 0.001
                cash -= shares_to_trade * S + trade_cost
                stock_position += shares_to_trade
                rehedged = True
                
                print(f"Day {day:2d}: Stock=${S:6.2f}, Delta={delta:.4f}, P&L=${pnl:+10,.2f}")
                print(f"         Rehedge: {'Buy' if shares_to_trade>0 else 'Sell'} "
                      f"{abs(shares_to_trade):6,.0f} shares (cost: ${trade_cost:.2f})")
            
            # Store results
            self.results.append({
                'day': day,
                'stock_price': S,
                'option_value': option_value,
                'delta': delta,
                'gamma': gamma,
                'portfolio_value': portfolio_value,
                'pnl': pnl,
                'rehedged': rehedged
            })
        
        # Final settlement
        final_S = stock_prices[-1]
        final_payoff = max(final_S - K, 0)
        final_option_value = final_payoff * contracts * 100
        final_stock_value = stock_position * final_S
        final_portfolio = cash + final_stock_value - final_option_value
        final_pnl = final_portfolio - self.initial_capital
        
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        
        print(f"\nStock Price Movement:")
        print(f"  Initial: ${stock_prices[0]:.2f}")
        print(f"  Final:   ${final_S:.2f}")
        print(f"  Change:  {(final_S/stock_prices[0]-1)*100:+.2f}%")
        
        print(f"\nOption Settlement:")
        print(f"  Premium Received:   ${premium_received:,.2f}")
        print(f"  Final Payoff:       ${final_option_value:,.2f}")
        print(f"  Option P&L:         ${premium_received - final_option_value:+,.2f}")
        
        print(f"\nPortfolio:")
        print(f"  Initial Capital:    ${self.initial_capital:,.2f}")
        print(f"  Final Value:        ${final_portfolio:,.2f}")
        print(f"  Total P&L:          ${final_pnl:+,.2f}")
        print(f"  Return:             {final_pnl/self.initial_capital*100:+.2f}%")
        
        # Calculate performance metrics
        pnls = [r['pnl'] for r in self.results]
        daily_returns = np.diff([0] + pnls)
        
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) 
                 if np.std(daily_returns) > 0 else 0)
        max_drawdown = min(pnls)
        
        print(f"\nPerformance Metrics:")
        print(f"  Sharpe Ratio:       {sharpe:.2f}")
        print(f"  Max Drawdown:       ${max_drawdown:,.2f}")
        print(f"  Win Days:           {sum(1 for r in daily_returns if r > 0)}/{len(daily_returns)}")
        
        print("\n" + "="*70)
        
        return self.results


def compare_strategies():
    """Compare different rehedging frequencies"""
    print("\n" + "="*70)
    print("STRATEGY COMPARISON: Different Rehedging Frequencies")
    print("="*70 + "\n")
    
    frequencies = [1, 3, 5, 10]
    comparison_results = []
    
    for freq in frequencies:
        print(f"\n{'─'*70}")
        print(f"Testing Rehedge Frequency: Every {freq} day(s)")
        print(f"{'─'*70}")
        
        backtest = DeltaHedgingBacktest(initial_capital=100000)
        results = backtest.run(
            S0=100, K=100, T=30/252, r=0.05, sigma=0.20,
            contracts=100, rehedge_freq=freq, num_days=30
        )
        
        final_pnl = results[-1]['pnl']
        pnls = [r['pnl'] for r in results]
        daily_returns = np.diff([0] + pnls)
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                 if np.std(daily_returns) > 0 else 0)
        
        comparison_results.append({
            'frequency': freq,
            'final_pnl': final_pnl,
            'sharpe': sharpe,
            'num_rehedges': sum(1 for r in results if r['rehedged'])
        })
    
    # Summary table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Rehedge Freq':>15} | {'# Rehedges':>12} | {'Final P&L':>15} | {'Sharpe':>10}")
    print("─"*70)
    
    for result in comparison_results:
        print(f"{result['frequency']:>13}d | {result['num_rehedges']:>12} | "
              f"${result['final_pnl']:>13,.2f} | {result['sharpe']:>10.2f}")
    
    print("\n" + "="*70)
    print("Key Insight: More frequent rehedging reduces risk but increases costs")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run single backtest
    backtest = DeltaHedgingBacktest(initial_capital=100000)
    results = backtest.run()
    
    # Compare different strategies
    print("\n" * 2)
    compare_strategies()
    
    print("\n✅ Backtesting demonstration complete!")
    print("\nThis shows:")
    print("  • Delta hedging implementation")
    print("  • Transaction cost modeling")
    print("  • Performance metrics calculation")
    print("  • Strategy parameter optimization")
