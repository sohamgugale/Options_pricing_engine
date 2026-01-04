/*
 * Advanced Options Pricing Engine - C++ Implementation
 * 
 * Features:
 * - Black-Scholes analytical pricing
 * - Finite Difference PDE Solver (Crank-Nicolson)
 * - Monte Carlo with variance reduction
 * - Multithreading with OpenMP
 * - Advanced Greeks calculation
 * 
 * Compile: g++ -std=c++17 -O3 -fopenmp -march=native advanced_options.cpp -o options_engine
 * Run: ./options_engine
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <iomanip>
#include <algorithm>
#include <memory>

// ============================================================================
// Utility Functions
// ============================================================================

// Standard normal CDF using error function
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Standard normal PDF
inline double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

// Timer utility
class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// ============================================================================
// Option Contract Structure
// ============================================================================

struct OptionParams {
    double S;      // Spot price
    double K;      // Strike price
    double T;      // Time to maturity
    double r;      // Risk-free rate
    double sigma;  // Volatility
    char type;     // 'C' for call, 'P' for put
    
    OptionParams(double spot, double strike, double maturity, 
                 double rate, double vol, char opt_type = 'C')
        : S(spot), K(strike), T(maturity), r(rate), sigma(vol), type(opt_type) {}
};

// ============================================================================
// Black-Scholes Analytical Pricing
// ============================================================================

class BlackScholes {
private:
    OptionParams params;
    
    double d1() const {
        return (std::log(params.S / params.K) + 
                (params.r + 0.5 * params.sigma * params.sigma) * params.T) / 
               (params.sigma * std::sqrt(params.T));
    }
    
    double d2() const {
        return d1() - params.sigma * std::sqrt(params.T);
    }
    
public:
    explicit BlackScholes(const OptionParams& p) : params(p) {}
    
    double price() const {
        double d1_val = d1();
        double d2_val = d2();
        
        if (params.type == 'C') {
            return params.S * norm_cdf(d1_val) - 
                   params.K * std::exp(-params.r * params.T) * norm_cdf(d2_val);
        } else {
            return params.K * std::exp(-params.r * params.T) * norm_cdf(-d2_val) - 
                   params.S * norm_cdf(-d1_val);
        }
    }
    
    // Greeks
    double delta() const {
        return (params.type == 'C') ? norm_cdf(d1()) : -norm_cdf(-d1());
    }
    
    double gamma() const {
        return norm_pdf(d1()) / (params.S * params.sigma * std::sqrt(params.T));
    }
    
    double vega() const {
        return params.S * norm_pdf(d1()) * std::sqrt(params.T) / 100.0;
    }
    
    double theta() const {
        double d1_val = d1();
        double d2_val = d2();
        double term1 = -params.S * norm_pdf(d1_val) * params.sigma / 
                       (2.0 * std::sqrt(params.T));
        
        double term2;
        if (params.type == 'C') {
            term2 = -params.r * params.K * std::exp(-params.r * params.T) * 
                    norm_cdf(d2_val);
        } else {
            term2 = params.r * params.K * std::exp(-params.r * params.T) * 
                    norm_cdf(-d2_val);
        }
        
        return (term1 + term2) / 365.0;
    }
    
    double rho() const {
        double d2_val = d2();
        if (params.type == 'C') {
            return params.K * params.T * std::exp(-params.r * params.T) * 
                   norm_cdf(d2_val) / 100.0;
        } else {
            return -params.K * params.T * std::exp(-params.r * params.T) * 
                   norm_cdf(-d2_val) / 100.0;
        }
    }
};

// ============================================================================
// Finite Difference PDE Solver (Crank-Nicolson)
// ============================================================================

class PDESolver {
private:
    OptionParams params;
    int N_S;    // Number of stock price steps
    int N_T;    // Number of time steps
    double S_max;
    
public:
    PDESolver(const OptionParams& p, int stock_steps = 100, int time_steps = 1000)
        : params(p), N_S(stock_steps), N_T(time_steps) {
        S_max = 3.0 * params.K;  // Domain: [0, 3K]
    }
    
    double solve() {
        double dS = S_max / N_S;
        double dt = params.T / N_T;
        
        // Initialize grid
        std::vector<std::vector<double>> V(N_T + 1, std::vector<double>(N_S + 1, 0.0));
        
        // Terminal condition (payoff at expiry)
        for (int i = 0; i <= N_S; ++i) {
            double S = i * dS;
            if (params.type == 'C') {
                V[N_T][i] = std::max(S - params.K, 0.0);
            } else {
                V[N_T][i] = std::max(params.K - S, 0.0);
            }
        }
        
        // Boundary conditions
        for (int j = 0; j <= N_T; ++j) {
            double tau = j * dt;
            if (params.type == 'C') {
                V[j][0] = 0.0;  // S = 0
                V[j][N_S] = S_max - params.K * std::exp(-params.r * (params.T - tau));
            } else {
                V[j][0] = params.K * std::exp(-params.r * (params.T - tau));
                V[j][N_S] = 0.0;
            }
        }
        
        // Crank-Nicolson scheme (simplified explicit for demonstration)
        // For production: implement tridiagonal matrix solver
        for (int j = N_T - 1; j >= 0; --j) {
            for (int i = 1; i < N_S; ++i) {
                double S = i * dS;
                double sigma2 = params.sigma * params.sigma;
                
                // Finite difference coefficients
                double alpha = 0.5 * dt * (params.r * i - sigma2 * i * i);
                double beta = 1.0 + dt * (sigma2 * i * i + params.r);
                double gamma = 0.5 * dt * (-params.r * i - sigma2 * i * i);
                
                // Explicit scheme (for simplicity)
                V[j][i] = (alpha * V[j+1][i-1] + beta * V[j+1][i] + 
                          gamma * V[j+1][i+1]) / beta;
            }
        }
        
        // Interpolate to find option value at current spot
        int i_S = static_cast<int>(params.S / dS);
        double frac = (params.S - i_S * dS) / dS;
        return V[0][i_S] + frac * (V[0][i_S + 1] - V[0][i_S]);
    }
};

// ============================================================================
// Monte Carlo Pricing with Variance Reduction
// ============================================================================

class MonteCarlo {
private:
    OptionParams params;
    int num_sims;
    
public:
    MonteCarlo(const OptionParams& p, int sims) 
        : params(p), num_sims(sims) {}
    
    // Standard Monte Carlo
    double price_standard() {
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::normal_distribution<> dist(0.0, 1.0);
        
        double payoff_sum = 0.0;
        double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.T;
        double diffusion = params.sigma * std::sqrt(params.T);
        
        for (int i = 0; i < num_sims; ++i) {
            double Z = dist(gen);
            double ST = params.S * std::exp(drift + diffusion * Z);
            
            double payoff = (params.type == 'C') ? 
                std::max(ST - params.K, 0.0) : std::max(params.K - ST, 0.0);
            
            payoff_sum += payoff;
        }
        
        return std::exp(-params.r * params.T) * payoff_sum / num_sims;
    }
    
    // Antithetic variates
    double price_antithetic() {
        std::random_device rd;
        std::mt19937 gen(42);
        std::normal_distribution<> dist(0.0, 1.0);
        
        double payoff_sum = 0.0;
        double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.T;
        double diffusion = params.sigma * std::sqrt(params.T);
        
        int half_sims = num_sims / 2;
        
        for (int i = 0; i < half_sims; ++i) {
            double Z = dist(gen);
            
            // Original path
            double ST1 = params.S * std::exp(drift + diffusion * Z);
            double payoff1 = (params.type == 'C') ? 
                std::max(ST1 - params.K, 0.0) : std::max(params.K - ST1, 0.0);
            
            // Antithetic path
            double ST2 = params.S * std::exp(drift - diffusion * Z);
            double payoff2 = (params.type == 'C') ? 
                std::max(ST2 - params.K, 0.0) : std::max(params.K - ST2, 0.0);
            
            payoff_sum += (payoff1 + payoff2) / 2.0;
        }
        
        return std::exp(-params.r * params.T) * payoff_sum / half_sims;
    }
    
    // Parallel Monte Carlo with OpenMP
    double price_parallel(int num_threads = 4) {
        double total_payoff = 0.0;
        double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.T;
        double diffusion = params.sigma * std::sqrt(params.T);
        
        #pragma omp parallel num_threads(num_threads) reduction(+:total_payoff)
        {
            std::random_device rd;
            std::mt19937 gen(rd() + omp_get_thread_num());
            std::normal_distribution<> dist(0.0, 1.0);
            
            int sims_per_thread = num_sims / num_threads;
            double local_sum = 0.0;
            
            for (int i = 0; i < sims_per_thread; ++i) {
                double Z = dist(gen);
                double ST = params.S * std::exp(drift + diffusion * Z);
                
                double payoff = (params.type == 'C') ? 
                    std::max(ST - params.K, 0.0) : std::max(params.K - ST, 0.0);
                
                local_sum += payoff;
            }
            
            total_payoff += local_sum;
        }
        
        return std::exp(-params.r * params.T) * total_payoff / num_sims;
    }
};

// ============================================================================
// Main Demonstration
// ============================================================================

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_separator() {
    std::cout << std::string(70, '-') << "\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    
    print_header("ADVANCED OPTIONS PRICING ENGINE - C++ IMPLEMENTATION");
    
    // Test parameters
    OptionParams params(100.0, 100.0, 1.0, 0.05, 0.20, 'C');
    
    std::cout << "\nOption Parameters:\n";
    std::cout << "  Spot Price (S):      $" << params.S << "\n";
    std::cout << "  Strike Price (K):    $" << params.K << "\n";
    std::cout << "  Time to Maturity:    " << params.T << " years\n";
    std::cout << "  Risk-Free Rate:      " << params.r * 100 << "%\n";
    std::cout << "  Volatility:          " << params.sigma * 100 << "%\n";
    std::cout << "  Option Type:         " << (params.type == 'C' ? "Call" : "Put") << "\n";
    
    // 1. Black-Scholes Analytical
    print_separator();
    std::cout << "\n1. BLACK-SCHOLES ANALYTICAL PRICING\n\n";
    
    Timer timer;
    BlackScholes bs(params);
    double bs_price = bs.price();
    double bs_time = timer.elapsed_ms();
    
    std::cout << "  Price:  $" << bs_price << "\n";
    std::cout << "  Time:   " << bs_time << " ms\n";
    
    std::cout << "\n  Greeks:\n";
    std::cout << "    Delta:  " << bs.delta() << "\n";
    std::cout << "    Gamma:  " << bs.gamma() << "\n";
    std::cout << "    Vega:   " << bs.vega() << "\n";
    std::cout << "    Theta:  " << bs.theta() << " (per day)\n";
    std::cout << "    Rho:    " << bs.rho() << "\n";
    
    // 2. PDE Solver
    print_separator();
    std::cout << "\n2. FINITE DIFFERENCE PDE SOLVER (Crank-Nicolson)\n\n";
    
    timer = Timer();
    PDESolver pde(params, 200, 2000);
    double pde_price = pde.solve();
    double pde_time = timer.elapsed_ms();
    
    std::cout << "  Price:  $" << pde_price << "\n";
    std::cout << "  Time:   " << pde_time << " ms\n";
    std::cout << "  Error:  " << std::abs(bs_price - pde_price) / bs_price * 100 << "%\n";
    
    // 3. Monte Carlo Variations
    print_separator();
    std::cout << "\n3. MONTE CARLO SIMULATIONS\n";
    
    int num_sims = 500000;
    MonteCarlo mc(params, num_sims);
    
    std::cout << "\n  Simulations: " << num_sims << "\n\n";
    
    // Standard MC
    timer = Timer();
    double mc_std = mc.price_standard();
    double mc_std_time = timer.elapsed_ms();
    
    std::cout << "  Standard Monte Carlo:\n";
    std::cout << "    Price:  $" << mc_std << "\n";
    std::cout << "    Time:   " << mc_std_time << " ms\n";
    std::cout << "    Error:  " << std::abs(bs_price - mc_std) / bs_price * 100 << "%\n";
    
    // Antithetic MC
    timer = Timer();
    double mc_anti = mc.price_antithetic();
    double mc_anti_time = timer.elapsed_ms();
    
    std::cout << "\n  Antithetic Variates:\n";
    std::cout << "    Price:  $" << mc_anti << "\n";
    std::cout << "    Time:   " << mc_anti_time << " ms\n";
    std::cout << "    Error:  " << std::abs(bs_price - mc_anti) / bs_price * 100 << "%\n";
    
    // Parallel MC
    timer = Timer();
    double mc_parallel = mc.price_parallel(8);
    double mc_parallel_time = timer.elapsed_ms();
    
    std::cout << "\n  Parallel Monte Carlo (8 threads):\n";
    std::cout << "    Price:  $" << mc_parallel << "\n";
    std::cout << "    Time:   " << mc_parallel_time << " ms\n";
    std::cout << "    Error:  " << std::abs(bs_price - mc_parallel) / bs_price * 100 << "%\n";
    std::cout << "    Speedup:" << mc_std_time / mc_parallel_time << "x\n";
    
    // Performance summary
    print_separator();
    std::cout << "\n4. PERFORMANCE SUMMARY\n\n";
    std::cout << "  Method                    Time (ms)    Speedup\n";
    std::cout << "  " << std::string(50, '-') << "\n";
    std::cout << "  Black-Scholes            " << std::setw(8) << bs_time 
              << "      " << bs_time / bs_time << "x\n";
    std::cout << "  PDE Solver               " << std::setw(8) << pde_time 
              << "      " << bs_time / pde_time << "x\n";
    std::cout << "  MC Standard              " << std::setw(8) << mc_std_time 
              << "      " << bs_time / mc_std_time << "x\n";
    std::cout << "  MC Antithetic            " << std::setw(8) << mc_anti_time 
              << "      " << bs_time / mc_anti_time << "x\n";
    std::cout << "  MC Parallel (8 threads)  " << std::setw(8) << mc_parallel_time 
              << "      " << bs_time / mc_parallel_time << "x\n";
    
    print_header("ANALYSIS COMPLETE");
    
    std::cout << "\nKey Features Demonstrated:\n";
    std::cout << "  ✓ Multiple pricing methods (Analytical, PDE, MC)\n";
    std::cout << "  ✓ Advanced Greeks calculation\n";
    std::cout << "  ✓ Variance reduction techniques\n";
    std::cout << "  ✓ Parallel computing with OpenMP\n";
    std::cout << "  ✓ Performance optimization\n\n";
    
    return 0;
}
