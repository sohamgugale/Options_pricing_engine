#include <vector>
#include <cmath>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

// --- UTILITY: Tridiagonal Matrix Algorithm (TDMA) ---
// Solves Ax = d.
// a: sub-diagonal (a[i] is coefficient for x[i-1] in row i)
// b: main diagonal
// c: super-diagonal (c[i] is coefficient for x[i+1] in row i)
vector<double> solve_tridiagonal(const vector<double>& a, const vector<double>& b, 
                                 const vector<double>& c, const vector<double>& d) {
    int n = d.size();
    vector<double> c_prime(n);
    vector<double> d_prime(n);
    vector<double> x(n);

    // 1. Forward Elimination
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {
        double temp = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / temp;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / temp;
    }

    // 2. Backward Substitution
    x[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    return x;
}

// --- ENGINE: Crank-Nicolson Solver ---
double price_american_option_cn(double S, double K, double T, double r, double sigma, 
                                int M, int N, bool is_call) {
    double S_max = 3 * K;      
    double dt = T / N;         
    double dS = S_max / M;     

    vector<double> V(M + 1);
    vector<double> S_grid(M + 1);

    // Initial Condition (Payoff at Maturity)
    for (int i = 0; i <= M; ++i) {
        S_grid[i] = i * dS;
        if (is_call) V[i] = max(0.0, S_grid[i] - K);
        else         V[i] = max(0.0, K - S_grid[i]);
    }

    // Matrix Coefficients vectors
    // System size is M-1 (internal nodes 1 to M-1)
    vector<double> a(M - 1, 0.0); // Sub-diagonal
    vector<double> b(M - 1, 0.0); // Main diagonal
    vector<double> c(M - 1, 0.0); // Super-diagonal

    // Time Stepping
    for (int j = N - 1; j >= 0; --j) {
        vector<double> d(M - 1); // RHS
        
        for (int i = 1; i < M; ++i) {
            // Map grid index 'i' to matrix row 'k'
            int k = i - 1;

            double sigma2 = sigma * sigma;
            double idx = (double)i;
            
            // Crank-Nicolson Coefficients
            double alpha = 0.25 * dt * (sigma2 * idx * idx - r * idx);
            double beta  = -0.5 * dt * (sigma2 * idx * idx + r);
            double gamma = 0.25 * dt * (sigma2 * idx * idx + r * idx);

            // Fill Matrix (Implicit Side)
            // Note: Inverting signs because moving to LHS: (I - 0.5*dt*L)
            // But standard CN formulation is usually: -alpha*V_{i-1} + (1-beta)*V_i - gamma*V_{i+1}
            // Let's stick to the stable definition:
            
            double a_val = -alpha;
            double b_val = 1.0 + dt * r - 2 * beta; // Wait, beta contains -0.5*dt... 
            // Let's simplify: 
            // beta is NEGATIVE. -2*beta is POSITIVE. 
            // So b_val = 1 + small_pos + pos. Diagonally dominant. Correct.
            double c_val = -gamma;

            // Store in vectors using Row Index 'k'
            b[k] = b_val;
            if (k > 0)     a[k] = a_val; // Use 'k' for sub-diagonal (fixes the bug!)
            if (k < M - 2) c[k] = c_val; // Use 'k' for super-diagonal

            // Fill RHS (Explicit Side: Previous Time Step)
            d[k] = alpha * V[i-1] + (1.0 - beta) * V[i] + gamma * V[i+1];
        }

        // Boundary Conditions (Dirichlet)
        // V[0] and V[M] are fixed/known boundary values
        // Adjust d[0] and d[M-2] to account for boundaries
        // V[0] = 0 (for Call) or K*exp(-rt) (for Put) - simplified to Payoff for American
        // V[M] = S_max - K (Call) or 0 (Put)
        
        // Actually, for American, boundaries are often just the payoff
        double val_0 = is_call ? 0.0 : (K - S_grid[0]); 
        double val_M = is_call ? (S_grid[M] - K) : 0.0;

        // Add boundary terms to RHS
        // Row k=0 (i=1) depends on V[0]. The term is -alpha*V[0] moved to RHS -> +alpha*V[0]
        // But we already added alpha*V[i-1] into d[k] from the explicit step? 
        // No, that was the explicit side. The Implicit side has terms like -alpha*V_{i-1}^{new}.
        // Since V_{i-1}^{new} is known (boundary), we move it to d[k].
        // a[0] (coeff for V[0]) is 0 in our matrix system, so we must add the value to d manually.
        
        double sigma2 = sigma * sigma;
        double alpha_1 = 0.25 * dt * (sigma2 * 1 * 1 - r * 1);
        d[0] += alpha_1 * val_0; 

        double idx_last = (double)(M-1);
        double gamma_last = 0.25 * dt * (sigma2 * idx_last * idx_last + r * idx_last);
        d[M-2] += gamma_last * val_M;

        // Solve
        vector<double> V_new = solve_tridiagonal(a, b, c, d);

        // Update Solution & Check Early Exercise
        for (int i = 1; i < M; ++i) {
            double payoff = is_call ? max(0.0, S_grid[i] - K) : max(0.0, K - S_grid[i]);
            V[i] = max(V_new[i-1], payoff); // American Condition
        }
    }

    int idx = (int)(S / dS);
    return V[idx] + (V[idx+1] - V[idx]) * (S - S_grid[idx]) / dS;
}

PYBIND11_MODULE(options_solver, m) {
    m.doc() = "C++ Finite Difference Solver";
    m.def("price_american", &price_american_option_cn, "Price American Option");
}
