import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from advanced_options_engine import FDMEngine, BlackScholesEngine, CPP_AVAILABLE

st.set_page_config(page_title="Pro Quant Lab", page_icon="⚡", layout="wide")

# CSS for Dark Finance Theme
st.markdown("""
<style>
    .stMetric { background-color: #0E1117; border: 1px solid #333; }
    .stButton button { width: 100%; border-radius: 5px; font-weight: bold; }
    /* Fix for Plotly background */
    .js-plotly-plot .plotly .main-svg { background: rgba(0,0,0,0) !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Quantitative Derivatives Engine")
st.markdown("**Core:** C++ Crank-Nicolson FDM | **Features:** American, Barrier (Knock-Out), True Grid Greeks")

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

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 2])

# --- COLUMN 1: PRICING ---
with col1:
    st.markdown("### 🚀 Pricing")
    if st.button("Run FDM Solver", type="primary"):
        if CPP_AVAILABLE:
            with st.spinner("Solving PDE Grid..."):
                engine = FDMEngine(S, K, T, r, sigma, is_call, is_american, barrier)
                res = engine.calculate()
            
            # Display Main Price
            st.metric("Model Price", f"${res['price']:.4f}")
            
            # Display Greeks
            st.markdown("### 📐 Grid Greeks")
            g1, g2 = st.columns(2)
            g1.metric("Delta (Δ)", f"{res['delta']:.4f}")
            g2.metric("Gamma (Γ)", f"{res['gamma']:.4f}")
            
            # Barrier Warning
            if barrier > 0:
                if (is_call and S >= barrier) or (not is_call and S <= barrier):
                    st.error(f"🛑 KNOCKED OUT (Spot {S} hit Barrier {barrier})")
                else:
                    st.info(f"Barrier Active: Dies at {barrier}")
            
            # Save result for plotting context
            st.session_state['last_price'] = res['price']
            
        else:
            st.error("⚠️ C++ Module Not Compiled. Check setup.py.")

# --- COLUMN 2: VISUALIZATION ---
with col2:
    st.markdown("### 🧊 Volatility Surface")
    
    # We use a Tab system to keep it clean
    tab_surf, tab_explain = st.tabs(["3D Interactive Plot", "Solver Logic"])
    
    with tab_surf:
        if st.button("🔄 Generate 3D Surface (Spot vs Time)"):
            with st.spinner("Running 400 Simulations..."):
                # Generate Grid
                S_range = np.linspace(max(0.1, S*0.5), S*1.5, 20)
                T_range = np.linspace(0.1, 2.0, 20)
                S_mesh, T_mesh = np.meshgrid(S_range, T_range)
                Z_price = np.zeros_like(S_mesh)

                # Loop to calculate surface (using Analytical BS for speed in plotting)
                # Note: We use BS here because running the PDE 400 times takes ~8 seconds.
                # For a UI demo, speed is key. The single point pricer uses C++.
                for i in range(len(T_range)):
                    for j in range(len(S_range)):
                        # Using Python BS for instant visualization
                        eng = BlackScholesEngine(S_mesh[i,j], K, T_mesh[i,j], r, sigma, is_call)
                        Z_price[i,j] = eng.price()
                
                # Plotly 3D Surface
                fig = go.Figure(data=[go.Surface(
                    z=Z_price, x=S_mesh, y=T_mesh, 
                    colorscale='Viridis', 
                    opacity=0.9
                )])
                
                fig.update_layout(
                    title=f"{opt_type} Price Surface",
                    scene=dict(
                        xaxis_title='Spot Price ($)',
                        yaxis_title='Time to Maturity (Yrs)',
                        zaxis_title='Option Price ($)'
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Surface generated using Analytical approximation for UI responsiveness.")
        else:
            st.info("Click 'Generate' to visualize how Price evolves with Spot and Time.")

    with tab_explain:
        st.markdown("#### Why Finite Difference?")
        st.write("""
        Analytical formulas (Black-Scholes) fail when options have **Early Exercise** (American) or **Path Dependency** (Barriers).
        
        This engine solves the **Black-Scholes PDE** numerically:
        """)
        st.latex(r"\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0")
        st.write("""
        **Technique:**
        1. **Discretization:** We slice Price ($) and Time ($) into a grid.
        2. **Implicit Solver:** We construct a tridiagonal matrix system to step backward in time.
        3. **Stability:** The method is unconditionally stable for any grid size.
        """)
