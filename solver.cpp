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
    if (n == 0) return {};
    
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

// --- ENGINE: Crank-Nicolson Solver (Robust Implementation) ---
double price_american_option_cn(double S, double K, double T, double r, double sigma, 
                                int M, int N, bool is_call) {
    double S_max = 4 * K;      // Extended boundary to avoid reflection errors
    double dt = T / N;         
    double dS = S_max / M;     

    vector<double> V(M + 1);
    vector<double> S_grid(M + 1);

    // 1. Initial Condition (Payoff at Maturity)
    for (int i = 0; i <= M; ++i) {
        S_grid[i] = i * dS;
        if (is_call) V[i] = max(0.0, S_grid[i] - K);
        else         V[i] = max(0.0, K - S_grid[i]);
    }

    // 2. Pre-compute Coefficients (A, B, C)
    // Formulation: -A*V_{i-1} + (1+B)*V_i - C*V_{i+1} = ...
    vector<double> A(M);
    vector<double> B(M);
    vector<double> C(M);

    for (int i = 1; i < M; ++i) {
        double i2 = (double)i * i;
        double sigma2 = sigma * sigma;
        
        A[i] = 0.25 * dt * (sigma2 * i2 - r * i);
        B[i] = 0.50 * dt * (sigma2 * i2 + r);
        C[i] = 0.25 * dt * (sigma2 * i2 + r * i);
    }

    // 3. Time Stepping
    vector<double> a(M - 1); // Sub-diagonal
    vector<double> b(M - 1); // Main diagonal
    vector<double> c(M - 1); // Super-diagonal
    vector<double> d(M - 1); // RHS

    for (int j = N - 1; j >= 0; --j) {
        
        // Build Matrix System for nodes 1 to M-1
        for (int i = 1; i < M; ++i) {
            int k = i - 1; // Matrix row index (0 to M-2)

            // LHS Matrix (Implicit)
            b[k] = 1.0 + B[i]; 
            if (k > 0)     a[k] = -A[i];
            if (k < M - 2) c[k] = -C[i];

            // RHS Vector (Explicit)
            // d[k] = A*V_{i-1} + (1-B)*V_i + C*V_{i+1} (using Old V)
            d[k] = A[i]*V[i-1] + (1.0 - B[i])*V[i] + C[i]*V[i+1];
        }

        // Apply Boundary Conditions to RHS
        // Node 0 and Node M are Dirichlet boundaries (Fixed)
        // We assume Boundary is constant over one time step for simplification
        
        // Lower Boundary (i=1 needs V[0])
        // LHS term was -A[1]*V[0]. Move to RHS -> +A[1]*V[0]
        // RHS term was +A[1]*V[0]. 
        // Total added to d[0]: 2 * A[1] * V[0]
        d[0] += 2.0 * A[1] * V[0];

        // Upper Boundary (i=M-1 needs V[M])
        // LHS term was -C[M-1]*V[M]. Move to RHS -> +C[M-1]*V[M]
        // RHS term was +C[M-1]*V[M].
        // Total added to d[M-2]: 2 * C[M-1] * V[M]
        d[M-2] += 2.0 * C[M-1] * V[M];

        // Solve Tridiagonal System
        vector<double> V_new = solve_tridiagonal(a, b, c, d);

        // Update & American Constraint
        for (int i = 1; i < M; ++i) {
            double payoff = is_call ? max(0.0, S_grid[i] - K) : max(0.0, K - S_grid[i]);
            V[i] = max(V_new[i-1], payoff); // Check early exercise
        }
    }

    // 4. Linear Interpolation for result
    int idx = (int)(S / dS);
    if (idx >= M) return V[M]; 
    return V[idx] + (V[idx+1] - V[idx]) * (S - S_grid[idx]) / dS;
}

PYBIND11_MODULE(options_solver, m) {
    m.doc() = "C++ Finite Difference Solver";
    m.def("price_american", &price_american_option_cn, "Price American Option");
}
