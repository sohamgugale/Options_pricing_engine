"""
Backtesting Dashboard Integration for Streamlit
Add this to options_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backtesting_framework import (
    DeltaHedgingStrategy, VolatilityArbitrageStrategy,
    Backtester, PerformanceMetrics
)

def add_backtesting_mode():
    """Add backtesting mode to the dashboard"""
    
    st.header("📊 Strategy Backtesting")
    
    st.markdown("""
    Test options trading strategies on historical data with realistic transaction costs
    and performance analytics.
    """)
    
    # Strategy selection
    strategy_type = st.selectbox(
        "Select Strategy",
        ["Delta Hedging", "Volatility Arbitrage", "Gamma Scalping"]
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
        
        if strategy_type == "Delta Hedging":
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
            if strategy_type == "Delta Hedging":
                strategy = DeltaHedgingStrategy(
                    initial_capital=initial_capital,
                    rehedge_threshold=rehedge_threshold,
                    rehedge_frequency=rehedge_freq
                )
            else:
                strategy = DeltaHedgingStrategy(initial_capital=initial_capital)
            
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
                    'Value': list(metrics.to_dict().values())
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
                
                # Drawdown chart
                st.markdown("---")
                st.subheader("📉 Drawdown Analysis")
                
                equity = results['Portfolio_Value'].values
                cumulative_max = pd.Series(equity).expanding().max()
                drawdown = (equity - cumulative_max) / cumulative_max * 100
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=results['Date'],
                    y=drawdown,
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color='red', width=0)
                ))
                
                fig_dd.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    hovermode='x unified',
                    height=300
                )
                
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Returns distribution
                st.markdown("---")
                st.subheader("📊 Returns Distribution")
                
                returns = results['Portfolio_Value'].pct_change().dropna() * 100
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Daily Returns',
                    marker_color='lightblue'
                ))
                
                fig_dist.update_layout(
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=300
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
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
                st.info("Try adjusting the date range or using a different ticker.")


# Add to main dashboard
# In options_dashboard.py, add to the mode selection:
"""
mode = st.selectbox(
    "Select Analysis Type",
    ["Single Option Pricing", "Greeks Analysis", "Volatility Surface", 
     "Portfolio Risk", "Model Comparison", "Market Data Analysis", 
     "Strategy Backtesting"]  # <-- ADD THIS
)

# Then at the bottom, add:
elif mode == "Strategy Backtesting":
    add_backtesting_mode()
"""
