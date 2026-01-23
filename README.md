# High-Performance Options Pricing Engine

[![Language](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.10-yellow.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)

### 🚀 [Live Demo](https://sohamgugale-options-pricing.streamlit.app) 
*(Note: Replace the URL above with your actual Streamlit link after deploying)*

## 💡 Overview
A professional-grade quantitative finance tool that bridges **Computational Mechanics** and **Derivatives Pricing**. 

Unlike standard "plug-and-play" Black-Scholes calculators, this engine implements a **Finite Difference Method (FDM)** solver from scratch in **C++**, capable of pricing **American Options** by solving the Parabolic PDE on a discretized grid.

## 🛠️ Technical Architecture
* **Core Engine (C++):** Implements the **Crank-Nicolson Scheme** (unconditionally stable) to solve the Black-Scholes Partial Differential Equation.
* **Solver:** Custom **Tridiagonal Matrix Algorithm (TDMA)** for O(N) linear system solving.
* **Integration:** Uses **PyBind11** to expose high-performance C++ binaries to Python as a compiled module.
* **Frontend:** Streamlit dashboard for real-time sensitivity analysis.

## 📊 Key Features
| Feature | Method | Complexity |
| :--- | :--- | :--- |
| **American Options** | Finite Difference (PDE) | O(M*N) |
| **European Options** | Analytical Black-Scholes | O(1) |
| **Grid Solver** | Crank-Nicolson (Implicit) | Unconditionally Stable |
| **Linear Algebra** | Thomas Algorithm | O(N) |

## 🔧 Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/sohamgugale/Options_pricing_engine.git
cd Options_pricing_engine
```

**2. Compile the C++ Engine**
```bash
pip install .
```

**3. Run the Dashboard**
```bash
streamlit run options_dashboard.py
```

---
*Built by **Soham Gugale** as a demonstration of Computational Mechanics applied to Quantitative Finance.*
