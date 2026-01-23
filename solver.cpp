#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

// --- UTILS ---
vector<double> solve_tridiagonal(const vector<double>& a, const vector<double>& b, 
                                 const vector<double>& c, const vector<double>& d) {
    int n = d.size();
    if (n == 0) return {};
    vector<double> c_prime(n), d_prime(n), x(n);

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {
        double temp = b[i] - a[i] * c_prime[i - 1];
        if (abs(temp) < 1e-9) temp = 1e-9;
        c_prime[i] = c[i] / temp;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / temp;
    }

    x[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    return x;
}

// --- CORE PDE SOLVER (Internal - Uses FIXED Grid) ---
vector<double> run_solver_on_fixed_grid(
    double S_min, double dS, int M, int N, // FIXED GRID PARAMS
    double K, double T, double r, double sigma, 
    double B_barrier, bool is_call, bool is_american) 
{
    double dt = T / N;         
    vector<double> V(M + 1);
    vector<double> S_grid(M + 1);

    // Initial Condition (t=T)
    for (int i = 0; i <= M; ++i) {
        S_grid[i] = S_min + i * dS;
        if (is_call) V[i] = max(0.0, S_grid[i] - K);
        else         V[i] = max(0.0, K - S_grid[i]);
        
        if (B_barrier > 0) {
            if (is_call && S_grid[i] >= B_barrier - 1e-7) V[i] = 0.0;
            if (!is_call && S_grid[i] <= B_barrier + 1e-7) V[i] = 0.0;
        }
    }

    vector<double> a(M - 1), b(M - 1), c(M - 1), d(M - 1);
    
    // Time Stepping
    for (int j = N - 1; j >= 0; --j) {
        for (int i = 1; i < M; ++i) {
            int k = i - 1;
            double Si = S_grid[i];
            double v1 = 0.5 * sigma * sigma * Si * Si * dt / (dS * dS);
            double v2 = 0.5 * r * Si * dt / dS;
            double v3 = r * dt;

            // Fully Implicit
            a[k] = -(v1 - v2);
            b[k] = 1.0 + 2*v1 + v3;
            c[k] = -(v1 + v2);
            d[k] = V[i];
        }

        double val_L = is_call ? 0.0 : (S_min <= B_barrier + 1e-9 && B_barrier>0 ? 0.0 : (K - S_min)*exp(-r*(T - j*dt)));
        double val_R = is_call ? (S_min + M*dS >= B_barrier - 1e-9 && B_barrier>0 ? 0.0 : ((S_min + M*dS) - K)*exp(-r*(T - j*dt))) : 0.0;

        d[0]   -= a[0] * val_L;
        d[M-2] -= c[M-2] * val_R;

        vector<double> V_new = solve_tridiagonal(a, b, c, d);
        
        V[0] = val_L;
        V[M] = val_R;
        for (int i = 1; i < M; ++i) {
            double cont = V_new[i-1];
            if (is_american) {
                double payoff = is_call ? max(0.0, S_grid[i] - K) : max(0.0, K - S_grid[i]);
                if (B_barrier > 0 && ((is_call && S_grid[i] >= B_barrier) || (!is_call && S_grid[i] <= B_barrier))) 
                    payoff = 0.0;
                V[i] = max(cont, payoff);
            } else {
                V[i] = cont;
            }
        }
    }
    return V;
}

// --- MAIN WRAPPER ---
map<string, double> solve_with_greeks(
    double S, double K, double T, double r, double sigma, 
    double B_barrier, int M, int N, bool is_call, bool is_american) 
{
    // 1. DEFINE GRID ONCE (The "Boundary Layer" Sizing)
    double vol_sqrt_t = sigma * sqrt(T);
    double S_max = K * exp(5.0 * vol_sqrt_t); 
    double S_min = 0.0;

    if (B_barrier > 0) {
        if (is_call) S_max = B_barrier;
        else         S_min = B_barrier;
    }
    if (S_max < K) S_max = K * 2.0;
    if (S_min > K) S_min = 0.0;

    double dS = (S_max - S_min) / M;     

    // 2. Base Run
    vector<double> V = run_solver_on_fixed_grid(S_min, dS, M, N, K, T, r, sigma, B_barrier, is_call, is_american);
    
    int idx = (int)((S - S_min) / dS);
    if (idx < 0 || idx >= M) return {{"price", 0.0}}; 
    
    double price = V[idx] + (V[idx+1] - V[idx]) * (S - (S_min + idx*dS)) / dS;
    double delta = (V[idx+1] - V[idx-1]) / (2 * dS);
    double gamma = (V[idx+1] - 2*V[idx] + V[idx-1]) / (dS * dS);

    // 3. Vega (Sensitivity to Sigma) - Uses SAME Grid
    double dSigma = 0.01;
    vector<double> V_vega = run_solver_on_fixed_grid(S_min, dS, M, N, K, T, r, sigma + dSigma, B_barrier, is_call, is_american);
    double price_vega = V_vega[idx] + (V_vega[idx+1] - V_vega[idx]) * (S - (S_min + idx*dS)) / dS;
    double vega = (price_vega - price) / dSigma;

    // 4. Rho (Sensitivity to Rate) - Uses SAME Grid
    double dr = 0.01;
    vector<double> V_rho = run_solver_on_fixed_grid(S_min, dS, M, N, K, T, r + dr, sigma, B_barrier, is_call, is_american);
    double price_rho = V_rho[idx] + (V_rho[idx+1] - V_rho[idx]) * (S - (S_min + idx*dS)) / dS;
    double rho = (price_rho - price) / dr;

    // 5. Theta (Sensitivity to Time)
    // Here we reduce T, but keep the SPACE grid (S_min, dS) same.
    double dT_bump = 1.0/365.0; 
    double theta = 0.0;
    if (T > dT_bump) {
         vector<double> V_theta = run_solver_on_fixed_grid(S_min, dS, M, N, K, T - dT_bump, r, sigma, B_barrier, is_call, is_american);
         double price_theta = V_theta[idx] + (V_theta[idx+1] - V_theta[idx]) * (S - (S_min + idx*dS)) / dS;
         
         // FIX: Theta is Price(Tomorrow) - Price(Today). 
         // Since we calculated Price(T - 1 day), that IS "Tomorrow's Price" (less time to maturity).
         // So diff is P(T-dt) - P(T). 
         // Usually Theta is defined as "change per day passing".
         // If P(T=1yr) = 10, and P(T=0.99yr) = 9.9, we lost 0.1. Theta should be -0.1.
         theta = (price_theta - price); // Per day decay
    }

    return {
        {"price", price},
        {"delta", delta},
        {"gamma", gamma},
        {"vega", vega},   
        {"rho", rho},     
        {"theta", theta} 
    };
}

PYBIND11_MODULE(options_solver, m) {
    m.doc() = "C++ FDM Solver with Full Greeks";
    m.def("solve", &solve_with_greeks, "Price Option with Sensitivity Analysis");
}
