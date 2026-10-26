# Finite-Difference Options Pricing Engine (C++ / pybind11)

A Black-Scholes PDE solver written from scratch in C++ and exposed to Python with pybind11. It prices European, American and barrier options and returns the full set of Greeks. A Streamlit dashboard sits on top for interactive sensitivity analysis.

## Method

- **Discretization:** Crank-Nicolson time-stepping on a uniform asset-price grid. The scheme is second-order in time and space and unconditionally stable.
- **Linear solve:** each time step is a tridiagonal system, solved in O(M) with the Thomas algorithm.
- **American exercise:** after each step, the solution is projected onto the payoff, V = max(V, payoff).
- **Barriers:** knock-out options are handled by placing the grid boundary on the barrier.
- **Grid sizing:** the price domain spans ±5σ√T around the strike.
- **Greeks:**
  - Δ and Γ come from central differences on the solution grid.
  - Vega, ρ and Θ come from bump-and-reprice on the *same* grid, so discretization error cancels in the difference.
- **Bindings:** the core solver is C++ and is compiled into a Python module (`options_solver`) with pybind11.

## Results

Run `python validate.py` to reproduce these.

| Check | Reference | Result |
|---|---|---|
| European calls and puts, 90 cases (S ∈ 80–120, T ∈ 0.25–1, σ ∈ 0.15–0.40), 400×400 grid | Closed-form Black-Scholes | max error 0.28%, median 0.02% |
| American put (S = K = 100, T = 1, r = 5%, σ = 20%) | 5,000-step binomial tree | 6.084 vs 6.090 (0.10%) |
| Put-call parity | C − P = S − Ke^(−rT) | passes (unit test) |
| American premium | American put ≥ European put | passes (unit test) |
| Runtime: price + all Greeks, 200×200 grid | — | 1.7 ms (single core) |

## Build and run

Install the package. This compiles the C++ extension:

```bash
pip install .
```

Run the unit tests:

```bash
python -m unittest discover tests
```

Reproduce the results table:

```bash
python validate.py
```

Launch the dashboard:

```bash
streamlit run options_dashboard.py
```

## Layout

| File | Contents |
|---|---|
| `solver.cpp` | Crank-Nicolson solver, Thomas algorithm, Greeks, pybind11 module |
| `advanced_options_engine.py` | Python wrappers (`FDMEngine`, closed-form `BlackScholesEngine`) |
| `tests/test_pricing.py` | Convergence, put-call parity and American-premium tests |
| `validate.py` | Accuracy and timing benchmark behind the results table |
| `options_dashboard.py` | Streamlit front end |

## Next steps

- Replace the projection step with PSOR or a penalty method, to get the American constraint enforced inside each solve.
- Add Rannacher time-stepping (a few implicit steps at the start) to damp oscillations from the payoff kink.
- Use a non-uniform grid concentrated near the strike.
