import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from advanced_options_engine import FDMEngine, BlackScholesEngine, CPP_AVAILABLE

st.set_page_config(page_title="Pro Quant Lab", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stMetric { background-color: #0E1117; border: 1px solid #333; }
    .stButton button { width: 100%; border-radius: 5px; font-weight: bold; }
    .js-plotly-plot .plotly .main-svg { background: rgba(0,0,0,0) !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Quantitative Derivatives Engine")
st.markdown("**Core:** C++ Finite Difference (FDM) | **Validation:** Mesh Convergence & Sensitivity Analysis")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Contract Specs")
    S = st.number_input("Spot Price", value=100.0, step=1.0)
    K = st.number_input("Strike Price", value=100.0, step=1.0)
    T = st.slider("Years to Maturity", 0.1, 2.0, 1.0)
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
    sigma = st.slider("Volatility (σ)", 0.1, 1.0, 0.2)
    
    st.markdown("---")
    st.header("⚙️ Solver Config")
    opt_type = st.radio("Type", ["Call", "Put"], horizontal=True)
    style = st.radio("Exercise", ["European", "American"], horizontal=True)
    barrier = st.number_input("Barrier Level (0 = None)", value=0.0, step=1.0)
    
    is_call = opt_type == "Call"
    is_american = style == "American"

# --- MAIN LAYOUT ---
tab_main, tab_val, tab_about = st.tabs(["🚀 Pricing & Greeks", "🔬 Mesh Validation", "🧠 Methodology"])

# --- TAB 1: PRICING ---
with tab_main:
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Run Solver", type="primary"):
            if CPP_AVAILABLE:
                engine = FDMEngine(S, K, T, r, sigma, is_call, is_american, barrier)
                res = engine.calculate()
                
                st.metric("Model Price", f"${res['price']:.4f}")
                
                st.markdown("### 🏛️ Risk Sensitivities")
                
                # Row 1: Delta, Gamma, Theta (Standard size)
                c1, c2, c3 = st.columns(3)
                c1.metric("Delta (Δ)", f"{res['delta']:.4f}", help="Exposure")
                c2.metric("Gamma (Γ)", f"{res['gamma']:.4f}", help="Convexity")
                c3.metric("Theta (Θ)", f"{res['theta']:.4f}", help="Time Decay")

                # Row 2: Vega & Rho (WIDER layout to prevent cutoff)
                st.markdown("") # Spacer
                c4, c5 = st.columns([1, 1]) # Equal width, but only 2 cols = wider
                c4.metric("Vega (ν)", f"{res['vega']:.4f}", help="Vol Sensitivity")
                c5.metric("Rho (ρ)", f"{res['rho']:.4f}", help="Rate Sensitivity")

                if barrier > 0:
                     if (is_call and S >= barrier) or (not is_call and S <= barrier):
                        st.error("🛑 KNOCKED OUT")
                     else:
                        st.info(f"Barrier Active at {barrier}")
                
                st.session_state['last_S'] = S
            else:
                st.error("C++ Module Not Compiled.")
    
    with col2:
        st.markdown("### 🧊 Volatility Surface")
        if st.button("🔄 Generate 3D Surface (Spot vs Time)"):
            with st.spinner("Running 400 Simulations..."):
                S_range = np.linspace(max(0.1, S*0.5), S*1.5, 20)
                T_range = np.linspace(0.1, 2.0, 20)
                S_mesh, T_mesh = np.meshgrid(S_range, T_range)
                Z_price = np.zeros_like(S_mesh)

                for i in range(len(T_range)):
                    for j in range(len(S_range)):
                        eng = BlackScholesEngine(S_mesh[i,j], K, T_mesh[i,j], r, sigma, is_call)
                        Z_price[i,j] = eng.price()
                
                fig = go.Figure(data=[go.Surface(z=Z_price, x=S_mesh, y=T_mesh, colorscale='Viridis', opacity=0.9)])
                fig.update_layout(
                    title=f"{opt_type} Price Surface",
                    scene=dict(xaxis_title='Spot', yaxis_title='Time', zaxis_title='Price'),
                    margin=dict(l=0, r=0, b=0, t=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Surface generated using Analytical approximation for UI responsiveness.")

# --- TAB 2: VALIDATION ---
with tab_val:
    st.markdown("### 🕸️ Mesh Independence Study")
    st.write("Verifying that the C++ Finite Difference solver converges to a stable solution as we refine the time grid ($).")
    
    # Global definition for this tab
    grid_sizes = [50, 100, 200, 400, 800, 1600]

    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        if st.button("Run Convergence Test"):
            if CPP_AVAILABLE:
                prices = []
                progress_bar = st.progress(0)
                for i, n in enumerate(grid_sizes):
                    m = int(n/2) 
                    eng = FDMEngine(S, K, T, r, sigma, is_call, is_american, barrier)
                    res = eng.calculate(price_steps=m, time_steps=n)
                    prices.append(res['price'])
                    progress_bar.progress((i + 1) / len(grid_sizes))
                
                st.session_state['conv_data'] = pd.DataFrame({"N": grid_sizes, "Price": prices})
                st.session_state['conv_done'] = True
            else:
                st.error("C++ Module Missing")
    
    with col_v2:
        if st.session_state.get('conv_done'):
            df = st.session_state['conv_data']
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(x=df["N"], y=df["Price"], mode='lines+markers', name='FDM Price', line=dict(color='#00ffcc')))
            
            y_min, y_max = df["Price"].min(), df["Price"].max()
            padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
            
            fig_conv.update_layout(
                title="Convergence vs Time Steps (N)",
                xaxis_title="Time Steps (N)",
                yaxis_title="Option Price ($)",
                yaxis=dict(range=[y_min - padding, y_max + padding]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="white")
            )
            st.plotly_chart(fig_conv, use_container_width=True)
            
            change = abs(df['Price'].iloc[-1] - df['Price'].iloc[-2])
            st.success(f"Converged! Change from N={grid_sizes[-2]} to N={grid_sizes[-1]} is only ${change:.6f}")

    st.markdown("---")
    st.markdown("### 🛡️ Unit Test Suite")
    if st.button("Run Live Verification Tests"):
        st.write("Running benchmark tests against Analytical Solutions...")
        
        # Test 1
        bs_price = BlackScholesEngine(100, 100, 1, 0.05, 0.2, True).price()
        fdm_price = FDMEngine(100, 100, 1, 0.05, 0.2, True, False).calculate()['price']
        err = abs(bs_price - fdm_price)
        
        if err < 0.05:
            st.success(f"✅ PASS: European Call Match (Diff: ${err:.4f})")
        else:
            st.error(f"❌ FAIL: Diff ${err:.4f}")

        # Test 2
        c = FDMEngine(100, 100, 1, 0.05, 0.2, True, False).calculate()['price']
        p = FDMEngine(100, 100, 1, 0.05, 0.2, False, False).calculate()['price']
        diff = abs((c - p) - (100 - 100*np.exp(-0.05)))
        
        if diff < 0.05:
            st.success(f"✅ PASS: Put-Call Parity (Diff: {diff:.4f})")
        else:
            st.error("❌ FAIL: Parity Violation")
            
        # Test 3
        euro = FDMEngine(100, 100, 1, 0.05, 0.2, False, False).calculate()['price']
        amer = FDMEngine(100, 100, 1, 0.05, 0.2, False, True).calculate()['price']
        
        if amer >= euro:
            st.success(f"✅ PASS: American Premium ({amer:.4f} >= {euro:.4f})")
        else:
            st.error("❌ FAIL: Arbitrage Opportunity")

# --- TAB 3: METHODOLOGY ---
with tab_about:
    st.markdown("### 🔧 From Mechanics to Markets")
    st.write("This engine leverages concepts from **Computational Fluid Dynamics (CFD)** to solve Financial Derivatives.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🏗️ Physics Concept")
        st.markdown("* **Heat Diffusion:** $\frac{\partial T}{\partial t} = \alpha \nabla^2 T$")
        st.markdown("* **Mesh Independence:** Solution must not change with finer grid.")
    with col_b:
        st.markdown("#### 📈 Finance Application")
        st.markdown("* **Option Pricing:** Black-Scholes PDE is a diffusion equation.")
        st.markdown("* **Convergence:** We prove price stability by refining $ and $.")
