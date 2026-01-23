#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

// --- TRIDIAGONAL SOLVER ---
vector<double> solve_tridiagonal(const vector<double>& a, const vector<double>& b, 
                                 const vector<double>& c, const vector<double>& d) {
    int n = d.size();
    if (n == 0) return {};
    vector<double> c_prime(n), d_prime(n), x(n);

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {
        double temp = b[i] - a[i] * c_prime[i - 1];
        if (abs(temp) < 1e-9) temp = 1e-9; // Avoid division by zero
        c_prime[i] = c[i] / temp;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / temp;
    }

    x[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    return x;
}

// --- ENGINE ---
map<string, double> price_option_solver(
    double S, double K, double T, double r, double sigma, 
    double B_barrier, 
    int M, int N, bool is_call, bool is_american) 
{
    // 0. Safety Checks (Prevent SegFaults)
    if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0) {
        return {{"price", 0.0}, {"delta", 0.0}, {"gamma", 0.0}};
    }

    // 1. Barrier Logic (Up-and-Out Call / Down-and-Out Put)
    // For this demo, let's standardize:
    // If Barrier > 0, it is an "Out" barrier.
    // Call: Dies if S >= B (Up-and-Out)
    // Put: Dies if S <= B (Down-and-Out)
    
    // Immediate Knock-Out Checks
    if (B_barrier > 0) {
        if (is_call && S >= B_barrier) return {{"price", 0.0}, {"delta", 0.0}, {"gamma", 0.0}};
        if (!is_call && S <= B_barrier) return {{"price", 0.0}, {"delta", 0.0}, {"gamma", 0.0}};
    }

    // 2. Grid Setup
    double S_max = 3.0 * max(S, K);
    double S_min = 0.0;

    // Adjust Grid for Barrier
    if (B_barrier > 0) {
        if (is_call) S_max = B_barrier; // Grid goes from 0 to Barrier
        else         S_min = B_barrier; // Grid goes from Barrier to S_max
    }

    // Prevent grid collapse
    if (S_max <= S_min) S_max = S_min + 10.0;

    double dt = T / N;         
    double dS = (S_max - S_min) / M;     

    vector<double> V(M + 1);
    vector<double> S_grid(M + 1);

    // 3. Initial Condition
    for (int i = 0; i <= M; ++i) {
        S_grid[i] = S_min + i * dS;
        if (is_call) V[i] = max(0.0, S_grid[i] - K);
        else         V[i] = max(0.0, K - S_grid[i]);
        
        // Zero out at barrier (if barrier hit at T=0)
        if (B_barrier > 0) {
            if (is_call && S_grid[i] >= B_barrier - 1e-9) V[i] = 0.0;
            if (!is_call && S_grid[i] <= B_barrier + 1e-9) V[i] = 0.0;
        }
    }

    // 4. Coefficients
    vector<double> alpha(M), beta(M), gamma(M);
    for (int i = 1; i < M; ++i) {
        double Si = S_grid[i]; // Use actual S value, not just index
        double sigma2 = sigma * sigma;
        
        // Coefficients based on S_grid[i] instead of i*dS
        // This handles shifted grids (S_min > 0)
        
        // PDE: dV/dt + rS dV/dS + 0.5 sigma^2 S^2 d2V/dS2 - rV = 0
        // Discretized terms for Si
        
        double v1 = dt/(dS*dS) * 0.5 * sigma2 * Si * Si;
        double v2 = dt/(2*dS) * r * Si;
        
        alpha[i] = v1 - v2; // Lower Diag (i-1)
        beta[i]  = -2*v1 - dt*r - 1; // Main Diag (i) - Note: This is for LHS implicit
        gamma[i] = v1 + v2; // Upper Diag (i+1)
        
        // Re-aligning to standard Crank Nicolson or Fully Implicit
        // Let's use Fully Implicit for maximum stability against SegFaults
        // -alpha*V_{i-1} + (1 - beta)*V_i - gamma*V_{i+1} = V_old
        
        // Actually, stick to the previous CN but be careful
        // Let's use simpler Fully Implicit to guarantee no oscillations
    }
    
    // FULLY IMPLICIT SCHEME (Stable)
    vector<double> a(M - 1), b(M - 1), c(M - 1), d(M - 1);

    for (int j = N - 1; j >= 0; --j) {
        for (int i = 1; i < M; ++i) {
            int k = i - 1;
            double Si = S_grid[i];
            
            double v1 = 0.5 * sigma * sigma * Si * Si * dt / (dS * dS);
            double v2 = 0.5 * r * Si * dt / dS;
            double v3 = r * dt;

            // LHS: (1 + 2v1 + v3)V_i - (v1 - v2)V_{i-1} - (v1 + v2)V_{i+1} = V_old
            a[k] = -(v1 - v2);
            b[k] = 1.0 + 2*v1 + v3;
            c[k] = -(v1 + v2);
            d[k] = V[i]; // Previous time step value
        }

        // Boundary Conditions
        double val_L = 0.0; // Left Boundary
        double val_R = 0.0; // Right Boundary

        // Left Boundary (S_min)
        if (is_call) val_L = 0.0; 
        else val_L = (S_min <= B_barrier + 1e-9 && B_barrier > 0) ? 0.0 : (K - S_min)*exp(-r*(T - j*dt));

        // Right Boundary (S_max)
        if (is_call) val_R = (S_max >= B_barrier - 1e-9 && B_barrier > 0) ? 0.0 : (S_max - K)*exp(-r*(T - j*dt)); 
        else val_R = 0.0;

        d[0]   -= a[0] * val_L;
        d[M-2] -= c[M-2] * val_R;

        vector<double> V_new = solve_tridiagonal(a, b, c, d);
        
        V[0] = val_L;
        V[M] = val_R;
        for (int i = 1; i < M; ++i) {
            double cont = V_new[i-1];
            if (is_american) {
                double payoff = is_call ? max(0.0, S_grid[i] - K) : max(0.0, K - S_grid[i]);
                // Check barrier during exercise
                if (B_barrier > 0) {
                     if (is_call && S_grid[i] >= B_barrier) payoff = 0.0;
                     if (!is_call && S_grid[i] <= B_barrier) payoff = 0.0;
                }
                V[i] = max(cont, payoff);
            } else {
                V[i] = cont;
            }
        }
    }

    // 5. Interpolate Result
    int idx = (int)((S - S_min) / dS);
    if (idx < 0 || idx >= M) return {{"price", 0.0}, {"delta", 0.0}, {"gamma", 0.0}};

    double price = V[idx] + (V[idx+1] - V[idx]) * (S - S_grid[idx]) / dS;
    double delta = (V[idx+1] - V[idx-1]) / (2 * dS);
    double gamma_val = (V[idx+1] - 2*V[idx] + V[idx-1]) / (dS * dS);
    
    return {{"price", price}, {"delta", delta}, {"gamma", gamma_val}};
}

PYBIND11_MODULE(options_solver, m) {
    m.doc() = "C++ Finite Difference Solver";
    m.def("solve", &price_option_solver, "Price Option with Greeks");
}
