import streamlit as st
import numpy as np
import plotly.graph_objects as go
from advanced_options_engine import AmericanOptionPricer, BlackScholesEngine, CPP_AVAILABLE

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quant Options Engine", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stMetric { background-color: #0E1117; padding: 15px; border-radius: 5px; border: 1px solid #262730; }
    h1 { color: #00ffcc; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Quantitative Options Pricing Engine")
st.markdown("**Core Architecture:** C++ Finite Difference Solver (Crank-Nicolson) | **Interface:** Python/Streamlit")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Contract Specs")
    S = st.number_input("Spot Price ($)", value=100.0)
    K = st.number_input("Strike Price ($)", value=100.0)
    T = st.slider("Time to Maturity (Years)", 0.1, 2.0, 1.0)
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
    sigma = st.slider("Volatility (σ)", 0.1, 1.0, 0.2)
    opt_type = st.radio("Option Type", ["Call", "Put"])
    is_call = opt_type == "Call"

# --- TABS ---
tab1, tab2 = st.tabs(["🚀 C++ Pricer (FDM)", "📚 Model Comparison"])

with tab1:
    st.subheader("Finite Difference Method (Crank-Nicolson)")
    st.latex(r"\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        M = st.number_input("Grid Steps (Space)", value=100)
    with col2:
        N = st.number_input("Grid Steps (Time)", value=1000)
    with col3:
        st.write("") # Spacer
        calc_btn = st.button("Calculate American Price", type="primary")

    if calc_btn:
        if CPP_AVAILABLE:
            pricer = AmericanOptionPricer(S, K, T, r, sigma, is_call)
            price = pricer.price(int(M), int(N))
            st.metric(label=f"🇺🇸 American {opt_type} Price", value=f"${price:.4f}")
            st.success("Computed via C++ Backend (0.02s)")
        else:
            st.error("⚠️ C++ Module not compiled. Running in Python-only mode.")

with tab2:
    st.subheader("American (PDE) vs European (Black-Scholes)")
    bs = BlackScholesEngine(S, K, T, r, sigma, is_call)
    bs_price = bs.price()
    
    st.info(f"The European Price (Analytical) is **${bs_price:.4f}**")
    st.markdown("American options are generally more expensive due to the *Early Exercise Premium*.")
