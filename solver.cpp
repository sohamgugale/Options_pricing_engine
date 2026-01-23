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

// --- CORE PDE SOLVER (Internal Function) ---
// Returns the full price vector at t=0
vector<double> run_solver_internal(
    double S, double K, double T, double r, double sigma, 
    double B_barrier, int M, int N, bool is_call, bool is_american, 
    double& theta_out, double& S_min_out, double& dS_out) 
{
    // 1. Dynamic Grid Sizing (Mechanics: Diffusion Length)
    // Domain should cover ~5 standard deviations to avoid boundary errors
    double vol_sqrt_t = sigma * sqrt(T);
    double S_max = K * exp(5.0 * vol_sqrt_t); 
    double S_min = 0.0;

    // Adjust for Barrier
    if (B_barrier > 0) {
        if (is_call) S_max = B_barrier;
        else         S_min = B_barrier;
    }
    
    // Safety clamp
    if (S_max < K) S_max = K * 2.0;
    if (S_min > K) S_min = 0.0;

    S_min_out = S_min;
    double dt = T / N;         
    double dS = (S_max - S_min) / M;     
    dS_out = dS;

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

    // Coefficients
    vector<double> a(M - 1), b(M - 1), c(M - 1), d(M - 1);
    
    // Previous time step storage for Theta
    vector<double> V_prev_step = V; 

    // Time Stepping
    for (int j = N - 1; j >= 0; --j) {
        // Store V before update (at j+1)
        if (j == N - 1) V_prev_step = V; // t=T
        
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

        // Boundaries
        double val_L = is_call ? 0.0 : (S_min <= B_barrier + 1e-9 && B_barrier>0 ? 0.0 : (K - S_min)*exp(-r*(T - j*dt)));
        double val_R = is_call ? (S_max >= B_barrier - 1e-9 && B_barrier>0 ? 0.0 : (S_max - K)*exp(-r*(T - j*dt))) : 0.0;

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
        
        // Capture Theta near T=0 (Current Time)
        // Actually, Theta is dV/dt. 
        // V currently holds V(0). The previous loop held V(dt).
        // So V(dt) is V_new from the *second to last* step? 
        // Simpler: Just run the loop. We need V at t=0 and V at t=dt.
        if (j == 0) {
             // We are at the last step (going from dt to 0).
             // V_new is V(0). V (before update) was V(dt).
             // We can estimate theta roughly here, or use the Bump method for consistency.
        }
    }
    return V;
}

// --- MAIN WRAPPER WITH SENSITIVITY ANALYSIS ---
map<string, double> solve_with_greeks(
    double S, double K, double T, double r, double sigma, 
    double B_barrier, int M, int N, bool is_call, bool is_american) 
{
    // 1. Base Run
    double theta_dummy, S_min, dS;
    vector<double> V = run_solver_internal(S, K, T, r, sigma, B_barrier, M, N, is_call, is_american, theta_dummy, S_min, dS);
    
    // Interpolate Base Price
    int idx = (int)((S - S_min) / dS);
    if (idx < 0 || idx >= M) return {{"price", 0.0}}; // Safety
    
    double price = V[idx] + (V[idx+1] - V[idx]) * (S - (S_min + idx*dS)) / dS;
    double delta = (V[idx+1] - V[idx-1]) / (2 * dS);
    double gamma = (V[idx+1] - 2*V[idx] + V[idx-1]) / (dS * dS);

    // 2. Vega (Sensitivity to Sigma) - "Bump and Revalue"
    // Mechanics: dPrice/dSigma
    double dSigma = 0.01;
    double theta_dummy2, S_min2, dS2;
    vector<double> V_vega = run_solver_internal(S, K, T, r, sigma + dSigma, B_barrier, M, N, is_call, is_american, theta_dummy2, S_min2, dS2);
    
    // Interpolate Vega Price
    // Note: Grid might change slightly due to dynamic sizing! 
    // Ideally, fix grid for sensitivities. For now, assume interp handles it.
    int idx2 = (int)((S - S_min2) / dS2);
    double price_vega = V_vega[idx2] + (V_vega[idx2+1] - V_vega[idx2]) * (S - (S_min2 + idx2*dS2)) / dS2;
    double vega = (price_vega - price) / dSigma;

    // 3. Rho (Sensitivity to Rate)
    double dr = 0.01;
    vector<double> V_rho = run_solver_internal(S, K, T, r + dr, sigma, B_barrier, M, N, is_call, is_american, theta_dummy2, S_min2, dS2);
    int idx3 = (int)((S - S_min2) / dS2);
    double price_rho = V_rho[idx3] + (V_rho[idx3+1] - V_rho[idx3]) * (S - (S_min2 + idx3*dS2)) / dS2;
    double rho = (price_rho - price) / dr;

    // 4. Theta (Sensitivity to Time)
    // Run for T - dt
    double dT_bump = 1.0/365.0; // One day
    if (T > dT_bump) {
         vector<double> V_theta = run_solver_internal(S, K, T - dT_bump, r, sigma, B_barrier, M, N, is_call, is_american, theta_dummy2, S_min2, dS2);
         int idx4 = (int)((S - S_min2) / dS2);
         double price_theta = V_theta[idx4] + (V_theta[idx4+1] - V_theta[idx4]) * (S - (S_min2 + idx4*dS2)) / dS2;
         theta_dummy = (price_theta - price) / dT_bump;
    } else {
        theta_dummy = 0.0;
    }

    return {
        {"price", price},
        {"delta", delta},
        {"gamma", gamma},
        {"vega", vega},   // New!
        {"rho", rho},     // New!
        {"theta", theta_dummy} // New!
    };
}

PYBIND11_MODULE(options_solver, m) {
    m.doc() = "C++ FDM Solver with Full Greeks";
    m.def("solve", &solve_with_greeks, "Price Option with Sensitivity Analysis");
}
