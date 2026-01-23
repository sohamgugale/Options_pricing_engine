#include <vector>
#include <cmath>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

// --- UTILITY: Tridiagonal Matrix Algorithm (TDMA) ---
// Solves Ax = d efficiently in O(N)
vector<double> solve_tridiagonal(const vector<double>& a, const vector<double>& b, 
                                 const vector<double>& c, const vector<double>& d) {
    int n = d.size();
    vector<double> c_prime(n);
    vector<double> d_prime(n);
    vector<double> x(n);

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

// --- ENGINE: Crank-Nicolson Finite Difference Solver ---
// Solves the Black-Scholes PDE for American Options
// V_t + 0.5*sigma^2*S^2*V_ss + r*S*V_s - rV = 0
double price_american_option_cn(double S, double K, double T, double r, double sigma, 
                                int M, int N, bool is_call) {
    double S_max = 3 * K;      
    double dt = T / N;         
    double dS = S_max / M;     

    vector<double> V(M + 1);
    vector<double> S_grid(M + 1);

    // 1. Boundary & Initial Conditions (at Maturity)
    for (int i = 0; i <= M; ++i) {
        S_grid[i] = i * dS;
        if (is_call) V[i] = max(0.0, S_grid[i] - K);
        else         V[i] = max(0.0, K - S_grid[i]);
    }

    vector<double> a(M - 1), b(M - 1), c(M - 1);

    // 2. Time Stepping (Backward)
    for (int j = N - 1; j >= 0; --j) {
        vector<double> d(M - 1);
        
        for (int i = 1; i < M; ++i) {
            double sigma2 = sigma * sigma;
            double idx = (double)i;
            
            double alpha = 0.25 * dt * (sigma2 * idx * idx - r * idx);
            double beta  = -0.5 * dt * (sigma2 * idx * idx + r);
            double gamma = 0.25 * dt * (sigma2 * idx * idx + r * idx);

            if (i > 1) a[i-2] = -alpha;
            b[i-1] = 1 + dt * r - 2 * beta;
            if (i < M-1) c[i-1] = -gamma;

            d[i-1] = alpha * V[i-1] + (1 - beta) * V[i] + gamma * V[i+1];
        }

        vector<double> V_new = solve_tridiagonal(a, b, c, d);

        // 3. American Constraint Check (Early Exercise)
        for (int i = 1; i < M; ++i) {
            double payoff = is_call ? max(0.0, S_grid[i] - K) : max(0.0, K - S_grid[i]);
            V[i] = max(V_new[i-1], payoff); 
        }
    }

    // 4. Linear Interpolation for result
    int idx = (int)(S / dS);
    return V[idx] + (V[idx+1] - V[idx]) * (S - S_grid[idx]) / dS;
}

PYBIND11_MODULE(options_solver, m) {
    m.doc() = "C++ Finite Difference Solver for Options Pricing";
    m.def("price_american", &price_american_option_cn, "Price American Option using Crank-Nicolson");
}
