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
    /* Fix for Plotly background */
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
    barrier = st.number_input("Barrier Level (0 = None)", value=0.0, step=1.0, help="Knock-Out Barrier")
    
    is_call = opt_type == "Call"
    is_american = style == "American"

# --- MAIN LAYOUT (TABS DEFINITION) ---
# This must be outside the sidebar!
tab_main, tab_val, tab_about = st.tabs(["🚀 Pricing & Greeks", "🔬 Mesh Validation", "🧠 Methodology"])

# --- TAB 1: PRICING ---
with tab_main:
    col1, col2 = st.columns([1, 2])
    
    # Left Col: Control & Metrics
    with col1:
        if st.button("Run Solver", type="primary"):
            if CPP_AVAILABLE:
                engine = FDMEngine(S, K, T, r, sigma, is_call, is_american, barrier)
                res = engine.calculate()
                
                st.metric("Model Price", f"${res['price']:.4f}")
                
                # Full Greeks with "Trader" Context
                st.markdown("### 🏛️ Risk Sensitivities")
                c1, c2, c3 = st.columns(3)
                c1.metric("Delta (Δ)", f"{res['delta']:.4f}", help="Exposure: Dollar change in option price for  move in underlying.")
                c2.metric("Gamma (Γ)", f"{res['gamma']:.4f}", help="Convexity: How much Delta changes for  move in underlying.")
                c3.metric("Vega (ν)", f"{res['vega']:.4f}", help="Vol Risk: Dollar change for 1% change in volatility.")
                
                c4, c5 = st.columns(2)
                c4.metric("Theta (Θ)", f"{res['theta']:.4f}", help="Time Decay: Dollar loss per day holding the position.")
                c5.metric("Rho (ρ)", f"{res['rho']:.4f}", help="Rate Risk: Dollar change for 1% change in interest rates.")

                if barrier > 0:
                     if (is_call and S >= barrier) or (not is_call and S <= barrier):
                        st.error("🛑 KNOCKED OUT")
                     else:
                        st.info(f"Barrier Active at {barrier}")
                
                # Save for Plotly context
                st.session_state['last_S'] = S
            else:
                st.error("C++ Module Not Compiled.")
    
    # Right Col: 3D Plot
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
                        # Use BS for visualization speed
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

# --- TAB 2: VALIDATION (Mesh Convergence) ---
with tab_val:
    st.markdown("### 🕸️ Mesh Independence Study")
    st.write("In Computational Mechanics, we must prove the solution does not depend on the grid size. This chart compares the Price convergence as we refine the Time Grid (N).")
    
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        if st.button("Run Convergence Test"):
            if CPP_AVAILABLE:
                grid_sizes = [100, 200, 400, 800, 1600]
                prices = []
                
                progress_bar = st.progress(0)
                for i, n in enumerate(grid_sizes):
                    m = int(n/2) 
                    eng = FDMEngine(S, K, T, r, sigma, is_call, is_american, barrier)
                    res = eng.calculate(price_steps=m, time_steps=n)
                    prices.append(res['price'])
                    progress_bar.progress((i + 1) / len(grid_sizes))
                
                # Store results for plot
                st.session_state['conv_data'] = pd.DataFrame({"Time Steps (N)": grid_sizes, "Price": prices})
                st.session_state['conv_done'] = True
            else:
                st.error("C++ Module Missing")
    
    with col_v2:
        if st.session_state.get('conv_done'):
            st.line_chart(st.session_state['conv_data'], x="Time Steps (N)", y="Price")
            prices = st.session_state['conv_data']['Price'].tolist()
            convergence_err = abs(prices[-1] - prices[-2])
            st.success(f"Converged within ${convergence_err:.5f} between N=800 and N=1600")

    # --- LIVE UNIT TESTS SECTION ---
    st.markdown("---")
    st.markdown("### 🛡️ Unit Test Suite")
    if st.button("Run Live Verification Tests"):
        st.write("Running benchmark tests against Analytical Solutions...")
        
        # Test 1: BS Convergence
        bs_price = BlackScholesEngine(100, 100, 1, 0.05, 0.2, True).price()
        fdm_price = FDMEngine(100, 100, 1, 0.05, 0.2, True, False).calculate()['price']
        err = abs(bs_price - fdm_price)
        
        if err < 0.05:
            st.success(f"✅ PASS: European Call Match (Diff: ${err:.4f})")
        else:
            st.error(f"❌ FAIL: European Call Mismatch (Diff: ${err:.4f})")

        # Test 2: Put-Call Parity
        c = FDMEngine(100, 100, 1, 0.05, 0.2, True, False).calculate()['price']
        p = FDMEngine(100, 100, 1, 0.05, 0.2, False, False).calculate()['price']
        parity_diff = abs((c - p) - (100 - 100*np.exp(-0.05)))
        
        if parity_diff < 0.05:
            st.success(f"✅ PASS: Put-Call Parity (Diff: {parity_diff:.4f})")
        else:
            st.error(f"❌ FAIL: Parity Violation")
            
        # Test 3: American Premium
        euro_put = FDMEngine(100, 100, 1, 0.05, 0.2, False, False).calculate()['price']
        amer_put = FDMEngine(100, 100, 1, 0.05, 0.2, False, True).calculate()['price']
        
        if amer_put >= euro_put:
             st.success(f"✅ PASS: American Premium ({amer_put:.4f} >= {euro_put:.4f})")
        else:
             st.error("❌ FAIL: Arbitrage Opportunity Detected")

# --- TAB 3: METHODOLOGY ---
with tab_about:
    st.markdown("### 🔧 From Mechanics to Markets")
    st.write("This engine leverages concepts from **Computational Fluid Dynamics (CFD)** to solve Financial Derivatives.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🏗️ Physics Concept")
        st.markdown("""
        * **Heat Diffusion:** $\frac{\partial T}{\partial t} = \alpha \nabla^2 T$
        * **Mesh Independence:** Solution must not change with finer grid.
        * **Material Sensitivity:** How $ changes if conductivity $ changes.
        """)
    with col_b:
        st.markdown("#### 📈 Finance Application")
        st.markdown("""
        * **Option Pricing:** Black-Scholes PDE is a diffusion equation.
        * **Convergence:** We prove price stability by refining $ and $.
        * **Greeks (Vega/Rho):** Calculated via Direct Differentiation (Bump-and-Revalue).
        """)

    st.markdown("---")
    st.markdown("#### 💻 Technical Implementation")
    st.code("""
    // C++ Core (Solver.cpp)
    // - Implements Thomas Algorithm for O(N) Tridiagonal Matrix Solving
    // - Uses Fully Implicit Scheme (Backward Euler) for unconditional stability
    // - Dynamic Grid Sizing based on Volatility Cones
    """, language="cpp")
