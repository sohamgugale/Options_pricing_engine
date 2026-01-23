import streamlit as st
import numpy as np
import plotly.graph_objects as go
from advanced_options_engine import FDMEngine, BlackScholesEngine, CPP_AVAILABLE

st.set_page_config(page_title="Pro Quant Lab", page_icon="⚡", layout="wide")

# CSS
st.markdown("""
<style>
    .stMetric { background-color: #0E1117; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Quantitative Derivatives Engine")
st.markdown("**Core:** C++ Crank-Nicolson FDM | **Features:** American, Barrier (Knock-Out), True Grid Greeks")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Contract Specs")
    S = st.number_input("Spot Price", 100.0)
    K = st.number_input("Strike Price", 100.0)
    T = st.slider("Years to Maturity", 0.1, 2.0, 1.0)
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
    sigma = st.slider("Volatility (σ)", 0.1, 1.0, 0.2)
    
    st.markdown("---")
    st.header("⚙️ Solver Config")
    opt_type = st.radio("Type", ["Call", "Put"])
    style = st.radio("Exercise", ["European", "American"])
    barrier = st.number_input("Barrier Level (0 = None)", value=0.0, help="Up-and-Out Barrier Level")
    
    is_call = opt_type == "Call"
    is_american = style == "American"

# --- MAIN PANEL ---
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🚀 Run FDM Solver", type="primary"):
        if CPP_AVAILABLE:
            engine = FDMEngine(S, K, T, r, sigma, is_call, is_american, barrier)
            res = engine.calculate()
            
            st.markdown("### 🎯 Pricing Results")
            st.metric("Model Price", f"${res['price']:.4f}")
            
            st.markdown("### 📐 Grid Greeks")
            c1, c2 = st.columns(2)
            c1.metric("Delta (Δ)", f"{res['delta']:.4f}")
            c2.metric("Gamma (Γ)", f"{res['gamma']:.4f}")
            
            if barrier > 0:
                st.info(f"Barrier Active: Option dies if S >= {barrier}")
            
        else:
            st.error("C++ Module Not Compiled")

with col2:
    st.markdown("### 🧠 Solver Logic (Why FDM?)")
    st.markdown("""
    * **Greeks:** Calculated via Finite Difference directly on the PDE grid (not analytical).
    * **Exotics:** Supports **Barrier Options** (Knock-Out) where Analytical BS fails.
    * **Boundary:** Uses Linearity Condition ({SS}=0$) rather than hardcoded heuristics.
    """)
    
    # Placeholder for plot
    if 'res' in locals() and CPP_AVAILABLE:
        st.markdown("#### Volatility Surface Preview")
        st.caption("(Run full sweep to visualize)")

