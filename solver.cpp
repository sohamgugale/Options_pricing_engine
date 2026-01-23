#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

// --- TRIDIAGONAL SOLVER (Standard) ---
vector<double> solve_tridiagonal(const vector<double>& a, const vector<double>& b, 
                                 const vector<double>& c, const vector<double>& d) {
    int n = d.size();
    if (n == 0) return {};
    vector<double> c_prime(n), d_prime(n), x(n);

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {
        double temp = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / temp;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / temp;
    }

    x[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    return x;
}

// --- ENGINE: Crank-Nicolson with Greeks & Barriers ---
map<string, double> price_option_solver(
    double S, double K, double T, double r, double sigma, 
    double B_barrier, // New: Barrier Level (0 = no barrier)
    int M, int N, bool is_call, bool is_american) 
{
    // 1. Grid Setup
    double S_max = 3.0 * max(S, K); 
    if (B_barrier > 0) S_max = B_barrier; // If Knock-out, grid ends at barrier

    double dt = T / N;         
    double dS = S_max / M;     

    vector<double> V(M + 1);
    vector<double> S_grid(M + 1);

    // 2. Initial Condition (Payoff)
    for (int i = 0; i <= M; ++i) {
        S_grid[i] = i * dS;
        if (is_call) V[i] = max(0.0, S_grid[i] - K);
        else         V[i] = max(0.0, K - S_grid[i]);
    }

    // 3. Coefficients (Theta Scheme)
    // dV/dt + rS dV/dS + 0.5 sigma^2 S^2 d2V/dS2 - rV = 0
    vector<double> alpha(M), beta(M), gamma(M);
    for (int i = 1; i < M; ++i) {
        double i2 = (double)i * i;
        double sigma2 = sigma * sigma;
        
        alpha[i] = 0.25 * dt * (sigma2 * i2 - r * i); // Sub-diagonal
        beta[i]  = -0.5 * dt * (sigma2 * i2 + r);     // Main diagonal contribution
        gamma[i] = 0.25 * dt * (sigma2 * i2 + r * i); // Super-diagonal
    }

    // 4. Time Stepping
    vector<double> a(M - 1), b(M - 1), c(M - 1), d(M - 1);

    for (int j = N - 1; j >= 0; --j) {
        for (int i = 1; i < M; ++i) {
            int k = i - 1; 

            // Crank-Nicolson System construction
            // -alpha*V_{i-1} + (1 - beta)*V_i - gamma*V_{i+1} = ... (New Time)
            //  alpha*V_{i-1} + (1 + beta)*V_i + gamma*V_{i+1}     (Old Time)
            
            // Note: Signs depend on how you arrange LHS/RHS. 
            // Standard form: (I - 0.5*L)V_new = (I + 0.5*L)V_old
            
            a[k] = -alpha[i];
            b[k] = 1.0 + 0.5 * dt * (sigma * sigma * i * i + r); // Simplified main diag
            c[k] = -gamma[i];
            
            // RHS (Explicit part)
            d[k] = alpha[i]*V[i-1] + (1.0 - (0.5 * dt * (sigma * sigma * i * i + r)))*V[i] + gamma[i]*V[i+1];
        }

        // Boundary Conditions (Linearity: V_SS = 0 -> V_i = 2*V_{i-1} - V_{i-2})
        // This is more robust than fixed values.
        // For i=1 (k=0): Depends on V[0].
        // For i=M-1 (k=M-2): Depends on V[M].
        
        // Apply Barrier or Dirichlet
        double val_0 = 0.0; // V at S=0
        if (!is_call) val_0 = K * exp(-r * (T - j * dt)); // Put at 0 is PV(K)
        
        double val_M = 0.0; // V at S_max
        if (is_call && B_barrier <= 0) val_M = S_max - K * exp(-r * (T - j * dt));
        if (B_barrier > 0) val_M = 0.0; // Knock-out condition: Value is 0 at barrier

        // Fix RHS for boundaries
        d[0]   -= a[0] * val_0; // Move known V[0] to RHS
        d[M-2] -= c[M-2] * val_M; // Move known V[M] to RHS
        
        vector<double> V_new = solve_tridiagonal(a, b, c, d);

        // Update grid
        V[0] = val_0; 
        V[M] = val_M;
        for (int i = 1; i < M; ++i) {
            double continuation = V_new[i-1];
            if (is_american) {
                double payoff = is_call ? max(0.0, S_grid[i] - K) : max(0.0, K - S_grid[i]);
                V[i] = max(continuation, payoff);
            } else {
                V[i] = continuation;
            }
        }
    }

    // 5. Compute Greeks & Price (Interpolation)
    int idx = (int)(S / dS);
    
    // Price
    double price = V[idx] + (V[idx+1] - V[idx]) * (S - S_grid[idx]) / dS;
    
    // Delta (Central Difference from Grid)
    double delta = (V[idx+1] - V[idx-1]) / (2 * dS);
    
    // Gamma (Second Derivative from Grid)
    double gamma_val = (V[idx+1] - 2*V[idx] + V[idx-1]) / (dS * dS);

    // Theta (Finite Difference in Time - approximation)
    // Ideally we store previous time step, but for now we omit or approximate
    
    map<string, double> results;
    results["price"] = price;
    results["delta"] = delta;
    results["gamma"] = gamma_val;
    
    return results;
}

PYBIND11_MODULE(options_solver, m) {
    m.doc() = "C++ Finite Difference Solver with Greeks";
    m.def("solve", &price_option_solver, "Price Option with Greeks");
}
