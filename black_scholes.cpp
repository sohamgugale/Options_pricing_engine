#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <iomanip>

// Standard normal cumulative distribution function
double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Black-Scholes formula for European options
class BlackScholes {
public:
    double S;  // Stock price
    double K;  // Strike price
    double T;  // Time to maturity (years)
    double r;  // Risk-free rate
    double sigma;  // Volatility
    
    BlackScholes(double stock, double strike, double time, double rate, double vol)
        : S(stock), K(strike), T(time), r(rate), sigma(vol) {}
    
    double d1() {
        return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    }
    
    double d2() {
        return d1() - sigma * std::sqrt(T);
    }
    
    double call_price() {
        double d1_val = d1();
        double d2_val = d2();
        return S * norm_cdf(d1_val) - K * std::exp(-r * T) * norm_cdf(d2_val);
    }
    
    double put_price() {
        double d1_val = d1();
        double d2_val = d2();
        return K * std::exp(-r * T) * norm_cdf(-d2_val) - S * norm_cdf(-d1_val);
    }
    
    // Greeks calculation using finite differences
    double delta_call(double dS = 0.01) {
        BlackScholes up(S + dS, K, T, r, sigma);
        BlackScholes down(S - dS, K, T, r, sigma);
        return (up.call_price() - down.call_price()) / (2 * dS);
    }
    
    double gamma(double dS = 0.01) {
        BlackScholes up(S + dS, K, T, r, sigma);
        BlackScholes down(S - dS, K, T, r, sigma);
        BlackScholes mid(S, K, T, r, sigma);
        return (up.call_price() - 2 * mid.call_price() + down.call_price()) / (dS * dS);
    }
    
    double vega(double dSigma = 0.01) {
        BlackScholes up(S, K, T, r, sigma + dSigma);
        BlackScholes down(S, K, T, r, sigma - dSigma);
        return (up.call_price() - down.call_price()) / (2 * dSigma);
    }
    
    double theta_call(double dT = 1.0/365.0) {
        if (T <= dT) return 0.0;
        BlackScholes future(S, K, T - dT, r, sigma);
        return (future.call_price() - call_price()) / dT;
    }
};

// Monte Carlo simulation for option pricing
class MonteCarlo {
private:
    double S, K, T, r, sigma;
    int num_sims;
    
    double simulate_path() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, 1.0);
        
        double Z = d(gen);
        double ST = S * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
        return ST;
    }
    
public:
    MonteCarlo(double stock, double strike, double time, double rate, double vol, int sims)
        : S(stock), K(strike), T(time), r(rate), sigma(vol), num_sims(sims) {}
    
    double call_price() {
        double payoff_sum = 0.0;
        
        for (int i = 0; i < num_sims; ++i) {
            double ST = simulate_path();
            double payoff = std::max(ST - K, 0.0);
            payoff_sum += payoff;
        }
        
        double avg_payoff = payoff_sum / num_sims;
        return std::exp(-r * T) * avg_payoff;
    }
    
    // Multithreaded version
    double call_price_parallel(int num_threads = 4) {
        std::vector<std::thread> threads;
        std::vector<double> thread_results(num_threads, 0.0);
        int sims_per_thread = num_sims / num_threads;
        
        auto worker = [this, sims_per_thread](double& result) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0.0, 1.0);
            
            double local_sum = 0.0;
            for (int i = 0; i < sims_per_thread; ++i) {
                double Z = d(gen);
                double ST = S * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
                local_sum += std::max(ST - K, 0.0);
            }
            result = local_sum;
        };
        
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker, std::ref(thread_results[i]));
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        double total_sum = 0.0;
        for (double val : thread_results) {
            total_sum += val;
        }
        
        double avg_payoff = total_sum / num_sims;
        return std::exp(-r * T) * avg_payoff;
    }
};

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Options Pricing & Risk Analytics Engine ===" << std::endl << std::endl;
    
    // Example parameters
    double S = 100.0;    // Stock price
    double K = 100.0;    // Strike price
    double T = 1.0;      // 1 year to maturity
    double r = 0.05;     // 5% risk-free rate
    double sigma = 0.2;  // 20% volatility
    
    // Black-Scholes pricing
    std::cout << "--- Black-Scholes Pricing ---" << std::endl;
    BlackScholes bs(S, K, T, r, sigma);
    std::cout << "Call Price: $" << bs.call_price() << std::endl;
    std::cout << "Put Price:  $" << bs.put_price() << std::endl << std::endl;
    
    // Greeks
    std::cout << "--- Greeks ---" << std::endl;
    std::cout << "Delta: " << bs.delta_call() << std::endl;
    std::cout << "Gamma: " << bs.gamma() << std::endl;
    std::cout << "Vega:  " << bs.vega() << std::endl;
    std::cout << "Theta: " << bs.theta_call() << std::endl << std::endl;
    
    // Monte Carlo pricing
    std::cout << "--- Monte Carlo Simulation ---" << std::endl;
    int num_sims = 500000;
    MonteCarlo mc(S, K, T, r, sigma, num_sims);
    
    auto start = std::chrono::high_resolution_clock::now();
    double mc_price = mc.call_price();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Call Price (Single-threaded): $" << mc_price << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl << std::endl;
    
    // Parallel Monte Carlo
    start = std::chrono::high_resolution_clock::now();
    double mc_price_parallel = mc.call_price_parallel(4);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Call Price (Parallel 4 threads): $" << mc_price_parallel << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl << std::endl;
    
    // Validation
    std::cout << "--- Validation ---" << std::endl;
    double error = std::abs(bs.call_price() - mc_price_parallel) / bs.call_price() * 100;
    std::cout << "BS vs MC Error: " << error << "%" << std::endl;
    
    return 0;
}