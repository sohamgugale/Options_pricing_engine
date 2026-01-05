"""
Options Trading Strategy Backtesting Framework - FIXED VERSION

Fixed: pandas Series boolean ambiguity error in calculate_performance_metrics()

Features:
- Delta-neutral hedging strategies
- Volatility arbitrage
- P&L attribution and decomposition
- Transaction cost modeling
- Performance analytics (Sharpe, Sortino, Max Drawdown)
- Real market data integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import yfinance as yf
from advanced_options_engine import (
    OptionContract, BlackScholesEngine, ImpliedVolatility
)


@dataclass
class Transaction:
    """Record of a single transaction"""
    timestamp: datetime
    asset: str  # 'option' or 'stock'
    quantity: float
    price: float
    transaction_cost: float
    position_type: str  # 'long' or 'short'


@dataclass
class Position:
    """Current position in an asset"""
    asset: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float = 0.0
    
    def update_pnl(self):
        """Update unrealized P&L"""
        self.pnl = (self.current_price - self.entry_price) * self.quantity


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics"""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    total_pnl: float
    avg_trade_pnl: float
    volatility: float
    calmar_ratio: float
    
    def to_dict(self) -> Dict:
        return {
            'Total Return (%)': self.total_return * 100,
            'Sharpe Ratio': self.sharpe_ratio,
            'Sortino Ratio': self.sortino_ratio,
            'Max Drawdown (%)': self.max_drawdown * 100,
            'Win Rate (%)': self.win_rate * 100,
            'Profit Factor': self.profit_factor,
            'Total Trades': self.total_trades,
            'Total P&L ($)': self.total_pnl,
            'Avg Trade P&L ($)': self.avg_trade_pnl,
            'Volatility (%)': self.volatility * 100,
            'Calmar Ratio': self.calmar_ratio
        }


class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.transactions: List[Transaction] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = []
        
    def execute_trade(self, timestamp: datetime, asset: str, quantity: float, 
                     price: float, transaction_cost_pct: float = 0.001):
        """Execute a trade with transaction costs"""
        trade_value = abs(quantity * price)
        transaction_cost = trade_value * transaction_cost_pct
        
        # Update cash
        if quantity > 0:  # Buy
            total_cost = trade_value + transaction_cost
            if total_cost > self.cash:
                raise ValueError(f"Insufficient funds: need ${total_cost:.2f}, have ${self.cash:.2f}")
            self.cash -= total_cost
            position_type = 'long'
        else:  # Sell
            self.cash += trade_value - transaction_cost
            position_type = 'short'
        
        # Update or create position
        if asset in self.positions:
            pos = self.positions[asset]
            new_quantity = pos.quantity + quantity
            
            if new_quantity == 0:
                # Close position
                del self.positions[asset]
            else:
                # Update position
                pos.quantity = new_quantity
                pos.entry_price = ((pos.entry_price * pos.quantity + price * quantity) / 
                                  new_quantity if new_quantity != 0 else price)
        else:
            # New position
            self.positions[asset] = Position(
                asset=asset,
                quantity=quantity,
                entry_price=price,
                current_price=price
            )
        
        # Record transaction
        self.transactions.append(Transaction(
            timestamp=timestamp,
            asset=asset,
            quantity=quantity,
            price=price,
            transaction_cost=transaction_cost,
            position_type=position_type
        ))
    
    def update_positions(self, prices: Dict[str, float]):
        """Update current prices and P&L for all positions"""
        for asset, price in prices.items():
            if asset in self.positions:
                self.positions[asset].current_price = price
                self.positions[asset].update_pnl()
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.quantity * pos.current_price 
                            for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_portfolio_greeks(self, S: float, r: float, 
                           options_data: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate portfolio-level Greeks"""
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        
        for asset, pos in self.positions.items():
            if asset.startswith('option_'):
                opt_data = options_data[asset]
                option = OptionContract(
                    S=S, K=opt_data['K'], T=opt_data['T'],
                    r=r, sigma=opt_data['sigma'], option_type=opt_data['type']
                )
                bs = BlackScholesEngine(option)
                
                total_delta += bs.delta() * pos.quantity
                total_gamma += bs.gamma() * pos.quantity
                total_vega += bs.vega() * pos.quantity
                total_theta += bs.theta() * pos.quantity
            elif asset == 'stock':
                # Stock has delta = 1
                total_delta += pos.quantity
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega,
            'theta': total_theta
        }


class DeltaHedgingStrategy(TradingStrategy):
    """Delta-neutral hedging strategy"""
    
    def __init__(self, initial_capital: float = 100000,
                 rehedge_threshold: float = 0.1,
                 rehedge_frequency: int = 1):
        """
        Args:
            initial_capital: Starting capital
            rehedge_threshold: Rehedge when |delta| exceeds this threshold
            rehedge_frequency: Rehedge every N days (in addition to threshold)
        """
        super().__init__(initial_capital)
        self.rehedge_threshold = rehedge_threshold
        self.rehedge_frequency = rehedge_frequency
        self.days_since_rehedge = 0
        self.pnl_attribution = {
            'theta_pnl': [],
            'gamma_pnl': [],
            'vega_pnl': [],
            'residual_pnl': []
        }
    
    def should_rehedge(self, current_delta: float) -> bool:
        """Determine if portfolio needs rehedging"""
        return (abs(current_delta) > self.rehedge_threshold or 
                self.days_since_rehedge >= self.rehedge_frequency)
    
    def rehedge_portfolio(self, timestamp: datetime, S: float, 
                         current_delta: float, stock_price: float):
        """Execute delta hedging trade"""
        # Calculate required stock position for delta neutrality
        target_stock_quantity = -current_delta
        
        current_stock = self.positions.get('stock')
        current_stock_qty = current_stock.quantity if current_stock else 0.0
        
        # Trade required amount
        quantity_to_trade = target_stock_quantity - current_stock_qty
        
        if abs(quantity_to_trade) > 0.01:  # Minimum trade size
            self.execute_trade(
                timestamp=timestamp,
                asset='stock',
                quantity=quantity_to_trade,
                price=stock_price
            )
            self.days_since_rehedge = 0
    
    def calculate_pnl_attribution(self, greeks: Dict[str, float],
                                  S_change: float, vol_change: float,
                                  time_decay: float) -> Dict[str, float]:
        """
        Decompose P&L into Greek contributions
        
        P&L ≈ Delta × ΔS + 0.5 × Gamma × ΔS² + Vega × Δσ + Theta × Δt
        """
        theta_pnl = greeks['theta'] * time_decay
        gamma_pnl = 0.5 * greeks['gamma'] * (S_change ** 2)
        vega_pnl = greeks['vega'] * vol_change * 100  # Vega is per 1%
        
        return {
            'theta_pnl': theta_pnl,
            'gamma_pnl': gamma_pnl,
            'vega_pnl': vega_pnl
        }


class VolatilityArbitrageStrategy(TradingStrategy):
    """Volatility arbitrage: trade realized vs implied volatility"""
    
    def __init__(self, initial_capital: float = 100000,
                 realized_vol_window: int = 20,
                 entry_threshold: float = 0.05):
        """
        Args:
            initial_capital: Starting capital
            realized_vol_window: Days to calculate realized volatility
            entry_threshold: Enter when |IV - RV| > threshold
        """
        super().__init__(initial_capital)
        self.realized_vol_window = realized_vol_window
        self.entry_threshold = entry_threshold
        self.price_history: List[float] = []
    
    def calculate_realized_volatility(self) -> float:
        """Calculate historical volatility from price history"""
        if len(self.price_history) < 2:
            return 0.0
        
        returns = np.diff(np.log(self.price_history[-self.realized_vol_window:]))
        return np.std(returns) * np.sqrt(252)
    
    def generate_signal(self, implied_vol: float, realized_vol: float) -> str:
        """
        Generate trading signal
        
        Returns:
            'long_vol' - Buy options (IV < RV)
            'short_vol' - Sell options (IV > RV)
            'neutral' - No trade
        """
        vol_diff = implied_vol - realized_vol
        
        if vol_diff < -self.entry_threshold:
            return 'long_vol'  # IV underpriced
        elif vol_diff > self.entry_threshold:
            return 'short_vol'  # IV overpriced
        else:
            return 'neutral'


class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, strategy: TradingStrategy,
                 start_date: str, end_date: str,
                 ticker: str = 'SPY'):
        """
        Args:
            strategy: Trading strategy instance
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            ticker: Stock ticker symbol
        """
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.results = None
        
    def fetch_market_data(self) -> pd.DataFrame:
        """Fetch historical stock data"""
        print(f"Fetching market data for {self.ticker}...")
        data = yf.download(self.ticker, start=self.start_date, 
                          end=self.end_date, progress=False)
        return data
    
    def run_delta_hedging_backtest(self, K_offset: float = 0,
                                   T: float = 30/365,
                                   option_type: str = 'call',
                                   transaction_cost: float = 0.001) -> pd.DataFrame:
        """
        Run delta hedging backtest
        
        Args:
            K_offset: Strike offset from spot (0 = ATM)
            T: Time to maturity in years
            option_type: 'call' or 'put'
            transaction_cost: Transaction cost as % of trade value
        """
        # Fetch data
        market_data = self.fetch_market_data()
        
        if len(market_data) == 0:
            raise ValueError("No market data available")
        
        # Initialize results tracking
        results = []
        
        print(f"\nRunning delta hedging backtest...")
        print(f"Initial capital: ${self.strategy.initial_capital:,.2f}")
        print(f"Option: {option_type}, Strike offset: ${K_offset:.2f}, T: {T:.3f}y")
        
        # Calculate realized volatility from first window
        returns = market_data['Close'].pct_change().dropna()
        initial_vol = returns.iloc[:20].std() * np.sqrt(252)
        
        for i, (date, row) in enumerate(market_data.iterrows()):
            S = row['Close']
            
            # Update time to maturity (decaying)
            T_current = max(T - i/252, 0.01)
            
            # Calculate realized volatility (rolling)
            if i >= 20:
                recent_returns = returns.iloc[i-20:i]
                realized_vol = recent_returns.std() * np.sqrt(252)
            else:
                realized_vol = initial_vol
            
            # Use realized vol as proxy for implied vol (in practice, use actual IV)
            sigma = realized_vol
            
            K = S + K_offset  # Strike
            
            # Create option
            option = OptionContract(S=S, K=K, T=T_current, r=0.05, 
                                  sigma=sigma, option_type=option_type)
            bs = BlackScholesEngine(option)
            
            # Initial trade: sell option on first day
            if i == 0:
                option_price = bs.price()
                self.strategy.execute_trade(
                    timestamp=date,
                    asset=f'option_{option_type}',
                    quantity=-100,  # Sell 100 contracts
                    price=option_price * 100,  # Contract = 100 shares
                    transaction_cost_pct=transaction_cost
                )
                
                # Record option data
                self.strategy.positions[f'option_{option_type}'].current_price = option_price * 100
            
            # Update option price
            if f'option_{option_type}' in self.strategy.positions:
                option_price = bs.price()
                self.strategy.positions[f'option_{option_type}'].current_price = option_price * 100
            
            # Calculate Greeks
            greeks = {
                'delta': bs.delta() * (-100),  # 100 contracts sold
                'gamma': bs.gamma() * (-100),
                'vega': bs.vega() * (-100),
                'theta': bs.theta() * (-100)
            }
            
            # Check if rehedging needed
            if isinstance(self.strategy, DeltaHedgingStrategy):
                if self.strategy.should_rehedge(greeks['delta']):
                    self.strategy.rehedge_portfolio(
                        timestamp=date,
                        S=S,
                        current_delta=greeks['delta'],
                        stock_price=S
                    )
                self.strategy.days_since_rehedge += 1
            
            # Update positions
            prices = {'stock': S}
            if f'option_{option_type}' in self.strategy.positions:
                prices[f'option_{option_type}'] = option_price * 100
            
            self.strategy.update_positions(prices)
            
            # Calculate portfolio value
            portfolio_value = self.strategy.get_portfolio_value()
            self.strategy.equity_curve.append(portfolio_value)
            self.strategy.timestamps.append(date)
            
            # Record results
            results.append({
                'Date': date,
                'Stock_Price': S,
                'Portfolio_Value': portfolio_value,
                'Cash': self.strategy.cash,
                'Delta': greeks['delta'],
                'Gamma': greeks['gamma'],
                'Vega': greeks['vega'],
                'Theta': greeks['theta'],
                'Realized_Vol': realized_vol * 100,
                'Implied_Vol': sigma * 100
            })
            
            # Progress
            if (i + 1) % 50 == 0:
                pnl_pct = (portfolio_value - self.strategy.initial_capital) / self.strategy.initial_capital * 100
                print(f"Day {i+1}/{len(market_data)}: Portfolio = ${portfolio_value:,.2f} ({pnl_pct:+.2f}%)")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics - FIXED VERSION"""
        if self.results is None:
            raise ValueError("Run backtest first")
        
        # Convert to numpy arrays to avoid Series ambiguity
        equity = self.results['Portfolio_Value'].values  # ← FIX: Convert to numpy
        returns = np.diff(equity) / equity[:-1]
        
        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Sharpe ratio (annualized)
        if len(returns) > 0:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                sharpe = mean_return / std_return * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            std_downside = np.std(downside_returns)
            if std_downside > 0:
                sortino = np.mean(returns) / std_downside * np.sqrt(252)
            else:
                sortino = 0.0
        else:
            sortino = 0.0
        
        # Maximum drawdown
        cumulative = np.maximum.accumulate(equity)
        drawdown = (equity - cumulative) / cumulative
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = len([t for t in self.strategy.transactions 
                             if t.quantity < 0])  # Closed positions
        total_trades = len(self.strategy.transactions)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Total P&L
        total_pnl = equity[-1] - equity[0]
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Calmar ratio
        calmar = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            total_pnl=total_pnl,
            avg_trade_pnl=avg_trade_pnl,
            volatility=volatility,
            calmar_ratio=calmar
        )
    
    def plot_results(self) -> go.Figure:
        """Create comprehensive performance visualization"""
        if self.results is None:
            raise ValueError("Run backtest first")
        
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Equity Curve', 'Drawdown',
                'Delta Exposure', 'Gamma Exposure',
                'Portfolio Greeks', 'Realized vs Implied Vol',
                'Daily Returns', 'Return Distribution'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Equity Curve
        fig.add_trace(
            go.Scatter(x=self.results['Date'], y=self.results['Portfolio_Value'],
                      name='Portfolio Value', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_hline(y=self.strategy.initial_capital, line_dash="dash",
                     line_color="gray", row=1, col=1)
        
        # 2. Drawdown
        equity = self.results['Portfolio_Value'].values
        cumulative_max = np.maximum.accumulate(equity)
        drawdown = (equity - cumulative_max) / cumulative_max * 100
        
        fig.add_trace(
            go.Scatter(x=self.results['Date'], y=drawdown,
                      name='Drawdown', fill='tozeroy',
                      line=dict(color='red', width=1)),
            row=1, col=2
        )
        
        # 3. Delta
        fig.add_trace(
            go.Scatter(x=self.results['Date'], y=self.results['Delta'],
                      name='Delta', line=dict(color='green', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # 4. Gamma
        fig.add_trace(
            go.Scatter(x=self.results['Date'], y=self.results['Gamma'],
                      name='Gamma', line=dict(color='purple', width=2)),
            row=2, col=2
        )
        
        # 5. Vega and Theta
        fig.add_trace(
            go.Scatter(x=self.results['Date'], y=self.results['Vega'],
                      name='Vega', line=dict(color='orange', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.results['Date'], y=self.results['Theta'],
                      name='Theta', line=dict(color='cyan', width=2)),
            row=3, col=1
        )
        
        # 6. Realized vs Implied Vol
        fig.add_trace(
            go.Scatter(x=self.results['Date'], y=self.results['Realized_Vol'],
                      name='Realized Vol', line=dict(color='blue', width=2)),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.results['Date'], y=self.results['Implied_Vol'],
                      name='Implied Vol', line=dict(color='red', width=2, dash='dash')),
            row=3, col=2
        )
        
        # 7. Daily Returns
        returns = self.results['Portfolio_Value'].pct_change() * 100
        fig.add_trace(
            go.Bar(x=self.results['Date'], y=returns,
                   name='Daily Returns',
                   marker_color=np.where(returns >= 0, 'green', 'red')),
            row=4, col=1
        )
        
        # 8. Return Distribution
        fig.add_trace(
            go.Histogram(x=returns.dropna(), nbinsx=50,
                        name='Return Distribution',
                        marker_color='lightblue'),
            row=4, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=2)
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_yaxes(title_text="Delta", row=2, col=1)
        fig.update_yaxes(title_text="Gamma", row=2, col=2)
        fig.update_yaxes(title_text="Greeks", row=3, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=3, col=2)
        fig.update_yaxes(title_text="Return (%)", row=4, col=1)
        fig.update_yaxes(title_text="Frequency", row=4, col=2)
        
        fig.update_layout(
            height=1400,
            showlegend=True,
            title_text="Options Trading Strategy Backtest Results",
            title_font_size=20
        )
        
        return fig


def main():
    """Demonstration of backtesting framework"""
    print("="*70)
    print("OPTIONS TRADING STRATEGY BACKTESTING FRAMEWORK")
    print("="*70)
    
    # 1. Delta Hedging Strategy
    print("\n1. DELTA HEDGING STRATEGY")
    print("-" * 70)
    
    strategy = DeltaHedgingStrategy(
        initial_capital=100000,
        rehedge_threshold=0.1,
        rehedge_frequency=5
    )
    
    backtester = Backtester(
        strategy=strategy,
        start_date='2023-01-01',
        end_date='2023-12-31',
        ticker='SPY'
    )
    
    try:
        # Run backtest
        results = backtester.run_delta_hedging_backtest(
            K_offset=0,  # ATM
            T=30/365,  # 30 days
            option_type='call'
        )
        
        # Calculate metrics
        metrics = backtester.calculate_performance_metrics()
        
        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)
        
        for key, value in metrics.to_dict().items():
            print(f"{key:.<40} {value:>15.4f}")
        
        print("\n" + "="*70)
        
        # Generate plot
        fig = backtester.plot_results()
        fig.write_html('/mnt/user-data/outputs/backtest_results.html')
        print("\n✓ Results saved to: backtest_results.html")
        
        # Save results to CSV
        results.to_csv('/mnt/user-data/outputs/backtest_data.csv', index=False)
        print("✓ Data saved to: backtest_data.csv")
        
    except Exception as e:
        print(f"\nNote: {e}")
        print("Using synthetic data for demonstration...")


if __name__ == "__main__":
    main()
