"""
Options Trading Strategy Backtesting Framework - BROADCASTING FIX

Fixed: Array broadcasting error in returns calculation
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
    asset: str
    quantity: float
    price: float
    transaction_cost: float
    position_type: str


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
        
        if quantity > 0:
            total_cost = trade_value + transaction_cost
            if total_cost > self.cash:
                raise ValueError(f"Insufficient funds")
            self.cash -= total_cost
            position_type = 'long'
        else:
            self.cash += trade_value - transaction_cost
            position_type = 'short'
        
        if asset in self.positions:
            pos = self.positions[asset]
            new_quantity = pos.quantity + quantity
            
            if new_quantity == 0:
                del self.positions[asset]
            else:
                pos.quantity = new_quantity
                pos.entry_price = ((pos.entry_price * pos.quantity + price * quantity) / 
                                  new_quantity if new_quantity != 0 else price)
        else:
            self.positions[asset] = Position(
                asset=asset,
                quantity=quantity,
                entry_price=price,
                current_price=price
            )
        
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


class DeltaHedgingStrategy(TradingStrategy):
    """Delta-neutral hedging strategy"""
    
    def __init__(self, initial_capital: float = 100000,
                 rehedge_threshold: float = 0.1,
                 rehedge_frequency: int = 1):
        super().__init__(initial_capital)
        self.rehedge_threshold = rehedge_threshold
        self.rehedge_frequency = rehedge_frequency
        self.days_since_rehedge = 0
    
    def should_rehedge(self, current_delta: float) -> bool:
        """Determine if portfolio needs rehedging"""
        return (abs(current_delta) > self.rehedge_threshold or 
                self.days_since_rehedge >= self.rehedge_frequency)
    
    def rehedge_portfolio(self, timestamp: datetime, S: float, 
                         current_delta: float, stock_price: float):
        """Execute delta hedging trade"""
        target_stock_quantity = -current_delta
        
        current_stock = self.positions.get('stock')
        current_stock_qty = current_stock.quantity if current_stock else 0.0
        
        quantity_to_trade = target_stock_quantity - current_stock_qty
        
        if abs(quantity_to_trade) > 0.01:
            self.execute_trade(
                timestamp=timestamp,
                asset='stock',
                quantity=quantity_to_trade,
                price=stock_price
            )
            self.days_since_rehedge = 0


class VolatilityArbitrageStrategy(TradingStrategy):
    """Volatility arbitrage strategy"""
    
    def __init__(self, initial_capital: float = 100000,
                 realized_vol_window: int = 20,
                 entry_threshold: float = 0.05):
        super().__init__(initial_capital)
        self.realized_vol_window = realized_vol_window
        self.entry_threshold = entry_threshold
        self.price_history: List[float] = []


class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, strategy: TradingStrategy,
                 start_date: str, end_date: str,
                 ticker: str = 'SPY'):
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
        """Run delta hedging backtest - FIXED ARRAY BROADCASTING"""
        
        market_data = self.fetch_market_data()
        
        if len(market_data) == 0:
            raise ValueError("No market data available")
        
        results = []
        
        print(f"\nRunning delta hedging backtest...")
        print(f"Initial capital: ${self.strategy.initial_capital:,.2f}")
        print(f"Option: {option_type}, Strike offset: ${K_offset:.2f}, T: {T:.3f}y")
        
        # CRITICAL FIX: Properly extract Close prices as 1D array
        # Handle both single-ticker and multi-ticker DataFrames
        if isinstance(market_data['Close'], pd.DataFrame):
            # Multi-ticker case (shouldn't happen with single ticker, but be safe)
            close_prices = market_data['Close'].iloc[:, 0].values
        else:
            # Single-ticker case (normal)
            close_prices = market_data['Close'].values
        
        # Ensure 1D array
        close_prices = np.asarray(close_prices).flatten()
        
        # Calculate returns safely
        if len(close_prices) < 2:
            raise ValueError("Insufficient data points")
        
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # Initial volatility
        if len(returns) >= 20:
            initial_vol = float(np.std(returns[:20]) * np.sqrt(252))
        else:
            initial_vol = 0.2
        
        # Main backtest loop
        for i, (date, row) in enumerate(market_data.iterrows()):
            # Extract scalar values safely
            if isinstance(row['Close'], (pd.Series, np.ndarray)):
                S = float(row['Close'].iloc[0] if isinstance(row['Close'], pd.Series) else row['Close'][0])
            else:
                S = float(row['Close'])
            
            T_current = max(T - i/252, 0.01)
            
            # Calculate realized volatility
            if i >= 20 and i <= len(returns):
                recent_returns = returns[max(0, i-20):i]
                if len(recent_returns) > 0:
                    realized_vol = float(np.std(recent_returns) * np.sqrt(252))
                else:
                    realized_vol = initial_vol
            else:
                realized_vol = initial_vol
            
            sigma = float(realized_vol)
            K_current = float(S + K_offset)
            
            # Create option
            option = OptionContract(
                S=S, 
                K=K_current, 
                T=T_current, 
                r=0.05, 
                sigma=sigma, 
                option_type=option_type
            )
            bs = BlackScholesEngine(option)
            
            # Initial trade
            if i == 0:
                option_price = float(bs.price())
                self.strategy.execute_trade(
                    timestamp=date,
                    asset=f'option_{option_type}',
                    quantity=-100,
                    price=option_price * 100,
                    transaction_cost_pct=transaction_cost
                )
                self.strategy.positions[f'option_{option_type}'].current_price = option_price * 100
            
            # Update option price
            if f'option_{option_type}' in self.strategy.positions:
                option_price = float(bs.price())
                self.strategy.positions[f'option_{option_type}'].current_price = option_price * 100
            
            # Calculate Greeks
            greeks = {
                'delta': float(bs.delta() * (-100)),
                'gamma': float(bs.gamma() * (-100)),
                'vega': float(bs.vega() * (-100)),
                'theta': float(bs.theta() * (-100))
            }
            
            # Rehedging
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
            
            portfolio_value = self.strategy.get_portfolio_value()
            self.strategy.equity_curve.append(portfolio_value)
            self.strategy.timestamps.append(date)
            
            # Record results
            results.append({
                'Date': date,
                'Stock_Price': float(S),
                'Portfolio_Value': float(portfolio_value),
                'Cash': float(self.strategy.cash),
                'Delta': float(greeks['delta']),
                'Gamma': float(greeks['gamma']),
                'Vega': float(greeks['vega']),
                'Theta': float(greeks['theta']),
                'Realized_Vol': float(realized_vol * 100),
                'Implied_Vol': float(sigma * 100)
            })
            
            if (i + 1) % 50 == 0:
                pnl_pct = (portfolio_value - self.strategy.initial_capital) / self.strategy.initial_capital * 100
                print(f"Day {i+1}/{len(market_data)}: Portfolio = ${portfolio_value:,.2f} ({pnl_pct:+.2f}%)")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics"""
        if self.results is None:
            raise ValueError("Run backtest first")
        
        equity = self.results['Portfolio_Value'].values
        returns = np.diff(equity) / equity[:-1]
        
        total_return = float((equity[-1] - equity[0]) / equity[0])
        
        if len(returns) > 0:
            mean_return = float(np.mean(returns))
            std_return = float(np.std(returns))
            sharpe = float(mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        else:
            sharpe = 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            std_downside = float(np.std(downside_returns))
            sortino = float(mean_return / std_downside * np.sqrt(252)) if std_downside > 0 else 0.0
        else:
            sortino = 0.0
        
        cumulative = np.maximum.accumulate(equity)
        drawdown = (equity - cumulative) / cumulative
        max_drawdown = float(np.min(drawdown))
        
        winning_trades = len([t for t in self.strategy.transactions if t.quantity < 0])
        total_trades = len(self.strategy.transactions)
        win_rate = float(winning_trades / total_trades if total_trades > 0 else 0)
        
        gross_profit = float(np.sum(returns[returns > 0]))
        gross_loss = float(abs(np.sum(returns[returns < 0])))
        profit_factor = float(gross_profit / gross_loss if gross_loss > 0 else 0)
        
        total_pnl = float(equity[-1] - equity[0])
        avg_trade_pnl = float(total_pnl / total_trades if total_trades > 0 else 0)
        volatility = float(np.std(returns) * np.sqrt(252))
        calmar = float(abs(total_return / max_drawdown) if max_drawdown != 0 else 0)
        
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
