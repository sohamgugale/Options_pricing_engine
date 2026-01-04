"""
Interactive Options Analytics Dashboard
Streamlit-based web application for options pricing and risk management

Deploy: streamlit run options_dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from advanced_options_engine import (
    OptionContract, BlackScholesEngine, MonteCarloAdvanced,
    ImpliedVolatility, HestonModel, PortfolioRiskAnalytics,
    VolatilitySurface
)
from backtesting_framework import (
    DeltaHedgingStrategy, VolatilityArbitrageStrategy,
    Backtester, PerformanceMetrics
)
import yfinance as yf
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Options Analytics Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">⚡ Advanced Options Analytics Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Quantitative Finance Tools for Derivatives Pricing & Risk Management</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎯 Analysis Mode")
    mode = st.selectbox(
    "Select Analysis Type",
    ["Single Option Pricing", "Greeks Analysis", "Volatility Surface", 
     "Portfolio Risk", "Model Comparison", "Market Data Analysis",
     "Strategy Backtesting"]  
    )
    
    st.markdown("---")
    
    # Common parameters
    st.header("📊 Option Parameters")
    S = st.number_input("Spot Price ($)", value=100.0, min_value=1.0, step=1.0)
    K = st.number_input("Strike Price ($)", value=100.0, min_value=1.0, step=1.0)
    T = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01, max_value=10.0, step=0.1)
    r = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.1) / 100
    sigma = st.number_input("Volatility (%)", value=20.0, min_value=1.0, max_value=100.0, step=1.0) / 100
    option_type = st.selectbox("Option Type", ["call", "put"])
    
    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
    **Developer:** Soham Gugale  
    **Target Roles:** Quantitative Research, Trading Analytics  
    **Tech Stack:** Python, NumPy, SciPy, Streamlit  
    
    **Features:**
    - Black-Scholes & Monte Carlo
    - Implied Volatility Solver
    - Advanced Greeks
    - Heston Model
    - Portfolio Risk Analytics
    """)

# Create option contract
option = OptionContract(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)

# Mode: Single Option Pricing
if mode == "Single Option Pricing":
    st.header("💰 Option Pricing Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    # Black-Scholes
    bs = BlackScholesEngine(option)
    bs_price = bs.price()
    
    with col1:
        st.metric("Black-Scholes Price", f"${bs_price:.4f}")
        st.markdown(f"**Moneyness:** {S/K:.3f}")
    
    # Monte Carlo
    mc = MonteCarloAdvanced(option, num_sims=100000)
    mc_price, mc_error = mc.price_antithetic()
    
    with col2:
        st.metric("Monte Carlo Price", f"${mc_price:.4f}", 
                 delta=f"±${mc_error:.4f}")
        error_pct = abs(bs_price - mc_price) / bs_price * 100
        st.markdown(f"**Error:** {error_pct:.3f}%")
    
    with col3:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        time_value = bs_price - intrinsic
        st.metric("Time Value", f"${time_value:.4f}")
        st.metric("Intrinsic Value", f"${intrinsic:.4f}")
    
    # Greeks
    st.markdown("---")
    st.subheader("📈 Greeks (Risk Sensitivities)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Delta", f"{bs.delta():.4f}")
        st.caption("∂V/∂S: Price sensitivity to spot")
    
    with col2:
        st.metric("Gamma", f"{bs.gamma():.4f}")
        st.caption("∂²V/∂S²: Delta sensitivity")
    
    with col3:
        st.metric("Vega", f"{bs.vega():.4f}")
        st.caption("∂V/∂σ: Vol sensitivity (per 1%)")
    
    with col4:
        st.metric("Theta", f"{bs.theta():.4f}")
        st.caption("∂V/∂t: Time decay (per day)")
    
    # Price evolution chart
    st.markdown("---")
    st.subheader("📊 Option Price vs Spot Price")
    
    spot_range = np.linspace(S * 0.5, S * 1.5, 100)
    prices = []
    
    for spot in spot_range:
        temp_option = OptionContract(S=spot, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
        prices.append(BlackScholesEngine(temp_option).price())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=prices, mode='lines', 
                            name='Option Price', line=dict(width=3)))
    
    # Intrinsic value line
    if option_type == "call":
        intrinsic_line = np.maximum(spot_range - K, 0)
    else:
        intrinsic_line = np.maximum(K - spot_range, 0)
    
    fig.add_trace(go.Scatter(x=spot_range, y=intrinsic_line, mode='lines',
                            name='Intrinsic Value', line=dict(dash='dash')))
    
    fig.add_vline(x=K, line_dash="dot", annotation_text="Strike", line_color="red")
    fig.add_vline(x=S, line_dash="dot", annotation_text="Current Spot", line_color="green")
    
    fig.update_layout(
        title="Option Value Analysis",
        xaxis_title="Spot Price ($)",
        yaxis_title="Option Price ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Mode: Greeks Analysis
elif mode == "Greeks Analysis":
    st.header("📊 Greeks Surface Analysis")
    
    # Generate Greeks surface
    spot_range = np.linspace(S * 0.7, S * 1.3, 50)
    time_range = np.linspace(0.1, T, 50)
    
    greek_type = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta"])
    
    Z = np.zeros((len(time_range), len(spot_range)))
    
    for i, t in enumerate(time_range):
        for j, s in enumerate(spot_range):
            temp_option = OptionContract(S=s, K=K, T=t, r=r, sigma=sigma, option_type=option_type)
            bs = BlackScholesEngine(temp_option)
            
            if greek_type == "Delta":
                Z[i, j] = bs.delta()
            elif greek_type == "Gamma":
                Z[i, j] = bs.gamma()
            elif greek_type == "Vega":
                Z[i, j] = bs.vega()
            else:  # Theta
                Z[i, j] = bs.theta()
    
    # 3D Surface plot
    fig = go.Figure(data=[go.Surface(x=spot_range, y=time_range, z=Z, colorscale='Viridis')])
    
    fig.update_layout(
        title=f"{greek_type} Surface",
        scene=dict(
            xaxis_title="Spot Price ($)",
            yaxis_title="Time to Maturity (years)",
            zaxis_title=greek_type
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("Heatmap View")
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        x=spot_range,
        y=time_range,
        z=Z,
        colorscale='RdYlGn'
    ))
    
    fig_heatmap.update_layout(
        title=f"{greek_type} Heatmap",
        xaxis_title="Spot Price ($)",
        yaxis_title="Time to Maturity (years)",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Mode: Volatility Surface
elif mode == "Volatility Surface":
    st.header("🌐 Implied Volatility Surface")
    
    st.info("📌 This feature demonstrates IV surface construction from market data")
    
    # Create synthetic IV surface
    strikes = np.linspace(S * 0.8, S * 1.2, 20)
    maturities = np.linspace(0.25, 2.0, 20)
    
    K_grid, T_grid = np.meshgrid(strikes, maturities)
    IV_surface = np.zeros_like(K_grid)
    
    # Generate IV surface with volatility smile/skew
    for i in range(len(maturities)):
        for j in range(len(strikes)):
            moneyness = strikes[j] / S
            # Simple volatility smile model
            smile = 0.02 * (moneyness - 1)**2
            term_structure = 0.01 * np.sqrt(maturities[i])
            IV_surface[i, j] = sigma + smile + term_structure
    
    # 3D Surface
    fig = go.Figure(data=[go.Surface(
        x=K_grid, y=T_grid, z=IV_surface * 100,
        colorscale='Plasma'
    )])
    
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike Price ($)",
            yaxis_title="Time to Maturity (years)",
            zaxis_title="Implied Volatility (%)"
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # IV Smile at current maturity
    st.subheader("Volatility Smile")
    
    smile_strikes = strikes
    smile_ivs = IV_surface[10, :] * 100  # Middle maturity
    
    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(
        x=smile_strikes, y=smile_ivs,
        mode='lines+markers',
        name='IV Smile',
        line=dict(width=3)
    ))
    
    fig_smile.add_vline(x=S, line_dash="dot", annotation_text="ATM", line_color="red")
    
    fig_smile.update_layout(
        title=f"Implied Volatility Smile (T={T:.2f}y)",
        xaxis_title="Strike Price ($)",
        yaxis_title="Implied Volatility (%)",
        height=400
    )
    
    st.plotly_chart(fig_smile, use_container_width=True)

# Mode: Portfolio Risk
elif mode == "Portfolio Risk":
    st.header("💼 Portfolio Risk Analytics")
    
    st.markdown("### Portfolio Positions")
    
    # Create sample portfolio
    num_positions = st.slider("Number of Positions", 1, 5, 3)
    
    positions = []
    for i in range(num_positions):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pos_K = st.number_input(f"Strike #{i+1}", value=K + (i-1)*5, key=f"k{i}")
        with col2:
            pos_type = st.selectbox(f"Type #{i+1}", ["call", "put"], key=f"t{i}")
        with col3:
            quantity = st.number_input(f"Quantity #{i+1}", value=100.0, key=f"q{i}")
        with col4:
            pos_option = OptionContract(S=S, K=pos_K, T=T, r=r, sigma=sigma, option_type=pos_type)
            price = BlackScholesEngine(pos_option).price()
            st.metric("Price", f"${price:.2f}")
        
        positions.append({
            'option': pos_option,
            'quantity': quantity,
            'price': price,
            'type': pos_type
        })
    
    # Calculate portfolio Greeks
    st.markdown("---")
    st.subheader("📊 Portfolio Greeks")
    
    portfolio = PortfolioRiskAnalytics(positions)
    greeks = portfolio.calculate_portfolio_greeks()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Delta", f"{greeks['delta']:.2f}")
    with col2:
        st.metric("Portfolio Gamma", f"{greeks['gamma']:.4f}")
    with col3:
        st.metric("Portfolio Vega", f"{greeks['vega']:.2f}")
    with col4:
        st.metric("Portfolio Theta", f"${greeks['theta']:.2f}/day")
    
    # P&L simulation
    st.markdown("---")
    st.subheader("📈 P&L Simulation")
    
    spot_shock_range = np.linspace(S * 0.9, S * 1.1, 50)
    pnl = []
    
    for spot in spot_shock_range:
        position_pnl = 0
        for pos in positions:
            shocked_option = OptionContract(
                S=spot, K=pos['option'].K, T=T, r=r, sigma=sigma, 
                option_type=pos['option'].option_type
            )
            new_price = BlackScholesEngine(shocked_option).price()
            position_pnl += (new_price - pos['price']) * pos['quantity']
        pnl.append(position_pnl)
    
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=spot_shock_range, y=pnl,
        mode='lines',
        fill='tozeroy',
        line=dict(width=3)
    ))
    
    fig_pnl.add_vline(x=S, line_dash="dot", annotation_text="Current Spot", line_color="red")
    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig_pnl.update_layout(
        title="Portfolio P&L vs Spot Price Movement",
        xaxis_title="Spot Price ($)",
        yaxis_title="P&L ($)",
        height=400
    )
    
    st.plotly_chart(fig_pnl, use_container_width=True)

# Mode: Model Comparison
elif mode == "Model Comparison":
    st.header("🔬 Pricing Model Comparison")
    
    col1, col2 = st.columns(2)
    
    # Black-Scholes
    bs = BlackScholesEngine(option)
    bs_price = bs.price()
    
    with col1:
        st.subheader("Black-Scholes")
        st.metric("Price", f"${bs_price:.4f}")
        st.markdown("""
        **Assumptions:**
        - Constant volatility
        - Log-normal returns
        - No jumps
        - European exercise
        """)
    
    # Monte Carlo variations
    mc = MonteCarloAdvanced(option, num_sims=100000)
    mc_std, mc_std_err = mc.price_standard()
    mc_anti, mc_anti_err = mc.price_antithetic()
    mc_cv, mc_cv_err = mc.price_control_variate()
    
    with col2:
        st.subheader("Monte Carlo Methods")
        
        mc_data = pd.DataFrame({
            'Method': ['Standard', 'Antithetic', 'Control Variate'],
            'Price': [mc_std, mc_anti, mc_cv],
            'Std Error': [mc_std_err, mc_anti_err, mc_cv_err],
            'Error %': [
                abs(bs_price - mc_std) / bs_price * 100,
                abs(bs_price - mc_anti) / bs_price * 100,
                abs(bs_price - mc_cv) / bs_price * 100
            ]
        })
        
        st.dataframe(mc_data, use_container_width=True)
    
    # Heston Model
    st.markdown("---")
    st.subheader("Heston Stochastic Volatility Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        kappa = st.slider("Mean Reversion (κ)", 0.5, 5.0, 2.0, 0.1)
    with col2:
        theta = st.slider("Long-term Var (θ)", 0.01, 0.1, 0.04, 0.01)
    with col3:
        sigma_v = st.slider("Vol of Vol (σᵥ)", 0.1, 0.5, 0.3, 0.05)
    
    rho = st.slider("Correlation (ρ)", -1.0, 0.0, -0.7, 0.1)
    
    if st.button("Run Heston Simulation", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            heston = HestonModel(
                S0=S, K=K, T=T, r=r,
                v0=sigma**2, kappa=kappa, theta=theta,
                sigma_v=sigma_v, rho=rho
            )
            heston_price = heston.price_call_mc(num_sims=50000)
            
            st.success(f"Heston Price: ${heston_price:.4f}")
            st.info(f"BS Price: ${bs_price:.4f}")
            st.warning(f"Difference: ${abs(heston_price - bs_price):.4f}")

# Mode: Market Data Analysis
elif mode == "Market Data Analysis":
    st.header("📡 Real-Time Market Data Analysis")
    
    ticker = st.text_input("Enter Ticker Symbol", value="AAPL")
    
    if st.button("Fetch Market Data", type="primary"):
        try:
            with st.spinner(f"Fetching {ticker} options data..."):
                stock = yf.Ticker(ticker)
                
                # Current price
                current_price = stock.info.get('currentPrice', stock.info.get('regularMarketPrice'))
                
                st.success(f"{ticker} Current Price: ${current_price:.2f}")
                
                # Get options chain
                exp_dates = stock.options
                
                if len(exp_dates) > 0:
                    selected_exp = st.selectbox("Expiration Date", exp_dates[:5])
                    
                    options_chain = stock.option_chain(selected_exp)
                    calls = options_chain.calls
                    
                    # Calculate days to expiration
                    exp_date = datetime.strptime(selected_exp, '%Y-%m-%d')
                    days_to_exp = (exp_date - datetime.now()).days
                    T_market = max(days_to_exp / 365.0, 0.01)
                    
                    st.info(f"Days to Expiration: {days_to_exp} | Time (years): {T_market:.4f}")
                    
                    # Filter for liquid options
                    calls_filtered = calls[calls['volume'] > 0].copy()
                    
                    # Calculate theoretical prices
                    calls_filtered['modelPrice'] = calls_filtered.apply(
                        lambda row: BlackScholesEngine(OptionContract(
                            S=current_price,
                            K=row['strike'],
                            T=T_market,
                            r=0.05,
                            sigma=row['impliedVolatility'],
                            option_type='call'
                        )).price() if row['impliedVolatility'] > 0 else 0,
                        axis=1
                    )
                    
                    calls_filtered['pricingError'] = (
                        (calls_filtered['lastPrice'] - calls_filtered['modelPrice']) / 
                        calls_filtered['lastPrice'] * 100
                    )
                    
                    # Display results
                    st.subheader("Options Chain Analysis")
                    
                    display_df = calls_filtered[[
                        'strike', 'lastPrice', 'modelPrice', 'pricingError',
                        'impliedVolatility', 'volume', 'openInterest'
                    ]].head(10)
                    
                    display_df.columns = [
                        'Strike', 'Market Price', 'Model Price', 'Error %',
                        'IV', 'Volume', 'Open Interest'
                    ]
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # IV Skew
                    st.subheader("Implied Volatility Skew")
                    
                    fig_iv = go.Figure()
                    fig_iv.add_trace(go.Scatter(
                        x=calls_filtered['strike'],
                        y=calls_filtered['impliedVolatility'] * 100,
                        mode='markers+lines',
                        name='Implied Volatility'
                    ))
                    
                    fig_iv.add_vline(x=current_price, line_dash="dot", 
                                    annotation_text="Current Price", line_color="red")
                    
                    fig_iv.update_layout(
                        title=f"{ticker} Implied Volatility Skew",
                        xaxis_title="Strike Price ($)",
                        yaxis_title="Implied Volatility (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_iv, use_container_width=True)
                    
                else:
                    st.warning("No options data available for this ticker")
                    
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.info("Please check ticker symbol and try again")

# Mode: Strategy Backtesting
elif mode == "Strategy Backtesting":
    st.header("📊 Strategy Backtesting")
    
    st.markdown("""
    Test options trading strategies on historical data with realistic transaction costs
    and performance analytics.
    """)
    
    # Strategy selection
    strategy_type = st.selectbox(
        "Select Strategy",
        ["Delta Hedging", "Volatility Arbitrage"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Backtest Parameters")
        
        ticker = st.text_input("Ticker Symbol", value="SPY")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))
        initial_capital = st.number_input("Initial Capital ($)", 
                                         value=100000, min_value=10000, step=10000)
    
    with col2:
        st.subheader("Strategy Settings")
        
        rehedge_threshold = st.slider("Rehedge Threshold (Delta)", 
                                     0.01, 1.0, 0.1, 0.01)
        rehedge_freq = st.slider("Rehedge Frequency (days)", 1, 20, 5)
        transaction_cost = st.slider("Transaction Cost (%)", 
                                     0.0, 0.5, 0.1, 0.01) / 100
    
    # Option parameters
    st.subheader("Option Specification")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        K_offset = st.number_input("Strike Offset ($)", value=0.0, step=1.0)
    with col2:
        T_days = st.number_input("Time to Maturity (days)", value=30, min_value=1, max_value=365)
    with col3:
        option_type = st.selectbox("Option Type", ["call", "put"])
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary"):
        
        with st.spinner("Running backtest... This may take a minute."):
            
            # Initialize strategy
            strategy = DeltaHedgingStrategy(
                initial_capital=initial_capital,
                rehedge_threshold=rehedge_threshold,
                rehedge_frequency=rehedge_freq
            )
            
            # Initialize backtester
            backtester = Backtester(
                strategy=strategy,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                ticker=ticker
            )
            
            try:
                # Run backtest
                results = backtester.run_delta_hedging_backtest(
                    K_offset=K_offset,
                    T=T_days/365,
                    option_type=option_type,
                    transaction_cost=transaction_cost
                )
                
                # Calculate performance metrics
                metrics = backtester.calculate_performance_metrics()
                
                st.success("✅ Backtest completed successfully!")
                
                # Display performance metrics
                st.markdown("---")
                st.subheader("📈 Performance Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", 
                             f"{metrics.total_return*100:.2f}%",
                             delta=f"${metrics.total_pnl:,.2f}")
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{metrics.max_drawdown*100:.2f}%")
                with col4:
                    st.metric("Win Rate", f"{metrics.win_rate*100:.1f}%")
                
                # Detailed metrics
                st.markdown("---")
                st.subheader("📊 Detailed Metrics")
                
                metrics_df = pd.DataFrame({
                    'Metric': list(metrics.to_dict().keys()),
                    'Value': [f"{v:.4f}" for v in metrics.to_dict().values()]
                })
                
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # Equity curve
                st.markdown("---")
                st.subheader("💰 Equity Curve")
                
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=results['Date'],
                    y=results['Portfolio_Value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))
                
                fig_equity.add_hline(
                    y=initial_capital,
                    line_dash="dash",
                    annotation_text="Initial Capital",
                    line_color="gray"
                )
                
                fig_equity.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Greeks evolution
                st.markdown("---")
                st.subheader("🎯 Portfolio Greeks Over Time")
                
                fig_greeks = go.Figure()
                
                fig_greeks.add_trace(go.Scatter(
                    x=results['Date'], y=results['Delta'],
                    name='Delta', line=dict(width=2)
                ))
                fig_greeks.add_trace(go.Scatter(
                    x=results['Date'], y=results['Gamma'],
                    name='Gamma', line=dict(width=2)
                ))
                fig_greeks.add_trace(go.Scatter(
                    x=results['Date'], y=results['Theta'],
                    name='Theta', line=dict(width=2)
                ))
                
                fig_greeks.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Greek Value",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_greeks, use_container_width=True)
                
                # Download results
                st.markdown("---")
                
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Backtest Results (CSV)",
                    data=csv,
                    file_name=f"backtest_{ticker}_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
                st.info("Note: Market data fetching requires internet connection. Try adjusting the date range or using a different ticker.")
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <b>Advanced Options Analytics Engine</b> | Built with Python, NumPy, SciPy, Plotly, Streamlit<br>
 | © 2025 Soham Gugale
</div>
""", unsafe_allow_html=True)
