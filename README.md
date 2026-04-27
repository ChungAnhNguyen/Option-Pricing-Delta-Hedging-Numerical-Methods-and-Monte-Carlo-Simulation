# Option Pricing & Delta Hedging — Numerical Methods and Monte Carlo Simulation

> A research-grade implementation of derivative pricing techniques across five research notebooks, covering PDE-based finite difference schemes, Monte Carlo simulation with variance reduction, stochastic volatility modelling (Heston), and dynamic delta-hedging strategies.

---

## Overview

This repository documents a complete quantitative finance research project on the pricing and risk management of European options. The work progresses systematically from foundational PDE numerical analysis to advanced stochastic volatility models, combining rigorous mathematical derivation with high-performance Python implementations.

The primary objective is to price vanilla call/put options through multiple independent methodologies and reconcile their outputs — a workflow directly aligned with production quantitative research and front-office model validation.

---

## Repository Structure

```
.
├── recherche1.ipynb   # Explicit finite difference scheme (Black-Scholes PDE)
├── recherche2.ipynb   # Implicit finite difference scheme
├── recherche3.ipynb   # Crank-Nicolson scheme
├── recherche4.ipynb   # Monte Carlo — Euler-Maruyama + variance reduction
└── recherche5.ipynb   # Heston model — Monte Carlo + COS method + delta hedging
```

---

## Recherche 1 — Explicit Finite Difference Scheme

**Topic:** Solving the Black-Scholes PDE via an explicit (forward Euler) scheme in log-space.

**Mathematical framework:**

The Black-Scholes PDE in log-space $x = \log(S)$ reads:

$$\partial_t u + \frac{\sigma^2}{2}\partial^2_{xx} u + \left(r - \frac{\sigma^2}{2}\right)\partial_x u - r u = 0$$

with terminal condition $u(T, x) = (e^x - K)^+$.

**Key contributions:**
- **Domain localisation:** Derivation of a compact spatial domain $\mathcal{D} = [s_{\min}, s_{\max}]$ with probabilistic guarantee $\mathbb{Q}(s_t \in \mathcal{D}) \geq 1 - \alpha$ for all $t \in [0, T]$, using the quantile function of the log-normal distribution.
- **Explicit Euler discretisation:** Implementation of the forward-in-time finite difference update with Von Neumann stability analysis; identification of the CFL-type stability condition on the ratio $h/\delta^2$.
- **Boundary conditions:** ITM approximation $u(t, s_{\max}) \approx e^{s_{\max}} - e^{-r(T-t)}K$ and OTM Dirichlet condition $u(t, s_{\min}) = 0$.
- **Validation:** Comparison with the analytical Black-Scholes formula; study of convergence order in $h$ and $\delta$.

**Notable implementation:** Parallelised grid sweeps using `multiprocessing.Pool` and `ProcessPoolExecutor` to reduce wall-clock time on fine grids.

---

## Recherche 2 — Implicit Finite Difference Scheme

**Topic:** Replacing the explicit update by a fully implicit (backward Euler) scheme, eliminating the stability constraint.

**Discretisation:** For each time step $i$, the scheme solves an $(m-1) \times (m-1)$ tridiagonal linear system:

$$a\, u_{i,j-1} + b\, u_{i,j} + c\, u_{i,j+1} = u_{i+1,j}, \quad j = 1, \ldots, m-1$$

where the coefficients $a$, $b$, $c$ are derived from central differences in space combined with the backward Euler time update.

**Key contributions:**
- **Thomas algorithm:** Unconditionally stable $\mathcal{O}(m)$ tridiagonal solver, avoiding any matrix inversion overhead — a standard technique in numerical PDE for finance.
- **Stability analysis:** Proof that the implicit scheme is unconditionally stable for all $(h, \delta)$, in contrast to the explicit case.
- **Accuracy comparison:** Convergence study against Recherche 1 results and the closed-form price; first-order in time vs. second-order in space.

---

## Recherche 3 — Crank-Nicolson Scheme

**Topic:** The Crank-Nicolson (CN) scheme — a second-order-in-time method obtained by averaging the explicit and implicit operators.

**Discretisation:** The CN update couples consecutive time layers through:

$$\alpha\, u_{i,j-1} + \beta\, u_{i,j} + \gamma\, u_{i,j+1} = \tilde{u}_{i+1,j}$$
$$\tilde{u}_{i+1,j} = a\, u_{i+1,j-1} + b\, u_{i+1,j} + c\, u_{i+1,j+1}$$

yielding an $\mathcal{O}(h^2 + \delta^2)$ convergent scheme.

**Key contributions:**
- **Second-order convergence in time:** Empirical verification that the CN scheme achieves $\mathcal{O}(h^2)$ temporal accuracy, halving the number of time steps required to meet a given error tolerance compared to Recherche 1/Recherche 2.
- **Symmetric tridiagonal solve:** Exploitation of the CN matrix structure for an efficient right-hand-side precomputation step followed by a Thomas sweep.
- **Benchmark:** Full convergence table across $(h, \delta)$ grids, cross-validated against the Black-Scholes closed form and the explicit/implicit results.

**Summary — Recherche 1/2/3:** The three PDE schemes provide a complete numerical analysis benchmark: explicit (conditionally stable, O(h)), implicit (unconditionally stable, O(h)), and Crank-Nicolson (unconditionally stable, O(h²)). All three are reconciled against the closed-form Black-Scholes formula.

---

## Recherche 4 — Monte Carlo Simulation & Variance Reduction

**Topic:** Pricing European options via Monte Carlo simulation of an Euler-Maruyama discretisation of a time-dependent volatility Black-Scholes model.

**Model:** $dS_t = r S_t\, dt + \sigma(t) S_t\, dW_t$ with a piecewise-linear deterministic volatility term structure $\sigma(t)$.

**Key contributions:**

### Path simulation
- `single_step`: Euler-Maruyama update using `random.gauss()` for normally distributed increments.
- `generate_path`: Full trajectory simulation over $n$ time steps.

### Closed-form benchmark
Derivation and implementation of the modified Black-Scholes formula with integrated variance:

$$C_0 = S_0\, \Phi(d_1) - K e^{-rT}\, \Phi(d_2), \quad I_T = \int_0^T \sigma^2(t)\, dt$$

with $I_T$ computed analytically in closed form for the given piecewise-linear $\sigma$.

### Variance reduction — Antithetic variables
- `single_step_anti`: Simultaneous simulation of paired paths $(Z, -Z)$ sharing the same Brownian increments.
- `generate_paths_pair_anti` / `price_call_put_anti`: Full antithetic estimator achieving a significant reduction in estimator variance at zero additional simulation cost.
- **Empirical result:** Confidence intervals roughly halved in width compared to the standard estimator at equal sample count.

### Statistical output
Each pricing function returns both a point estimate and a $(1 - \alpha)$ confidence interval constructed via the CLT: $\hat{C} \pm z_{1-\alpha/2} \cdot \hat{\sigma}/\sqrt{N}$.

---

## Recherche 5 — Heston Stochastic Volatility: Monte Carlo, COS Method & Delta Hedging

**Topic:** Extension to the Heston stochastic volatility model; implementation of three independent pricing methods; simulation of a discrete delta-hedging strategy.

### 1. Heston Model

The asset price $(S_t)$ and instantaneous variance $(\nu_t)$ jointly satisfy:

$$\begin{cases} d\log S_t = \left(r - \frac{\nu_t}{2}\right)dt + \sqrt{\nu_t}\, d\widetilde{W}^{[S]}_t \\ d\nu_t = \kappa(\theta - \nu_t)\, dt + \xi\sqrt{\nu_t}\left(\rho\, d\widetilde{W}^{[S]}_t + \sqrt{1-\rho^2}\, d\widetilde{W}^{[\nu]}_t\right) \end{cases}$$

with Feller condition $2\kappa\theta > \xi^2$ ensuring $\nu_t > 0$ a.s.

**Cholesky decomposition:** Correlated Brownian drivers $(W^{[S]}, W^{[\nu]})$ are expressed in terms of two independent Gaussians, enabling direct simulation without rejection sampling.

### 2. Euler-Maruyama Discretisation with Variance Truncation

To prevent negative values under the square root, a **truncation scheme** $(\hat{\nu}_{t_i})^+ = \max(\hat{\nu}_{t_i}, 0)$ is applied at every time step. The `single_step` and `generate_path` functions implement this scheme efficiently.

### 3. Monte Carlo Pricing

`price_call_put` prices calls and puts under Heston via plain Monte Carlo, with CLT-based confidence intervals and optional antithetic extension.

### 4. COS Method (Fourier-Based Semi-Analytical Pricing)

Implementation of the Fang-Oosterlee COS formula exploiting the **analytical characteristic function** of the Heston log-price:

$$C_0 \approx e^{-rT} K \, \mathrm{Re}\!\left[\sum_{k=0}^{N-1} \varphi\!\left(\frac{k\pi}{b-a}, T; \nu_0\right) U_k \exp\!\left(ik\pi\frac{\log(S_0/K)-a}{b-a}\right)\right]$$

where $\varphi$ is the Heston characteristic function computed in closed form. This yields a near-exact reference price in $\mathcal{O}(N)$ operations, used to benchmark the Monte Carlo estimator.

**Key implementation details:**
- Modular design with separate functions for $D_1$, $g$, $\varphi$, $\chi_k$, $\psi_k$, $U_k$ using `cmath` for complex arithmetic.
- Domain parameters $a = -10\sqrt{\theta T}$, $b = 10\sqrt{\theta T}$ following Oosterlee & Grzelak.
- Convergence to MC estimate confirmed at $N = 128$ terms.

### 5. Delta Hedging Simulation

**Strategy:** Simulation of a discrete-time self-financing delta-hedging strategy over $N = 16{,}384$ paths for a short call position.

At each rebalancing date $t_i$:

$$\Delta_i = \frac{\partial u}{\partial S}(t_i, \widehat{S}_{t_i})$$

The hedging P&L $\Pi(T)$ is tracked path-by-path, decomposed into:
- Premium investment at the risk-free rate,
- Stock position rebalancing funded by borrowing,
- Final payoff settlement.

**Rebalancing frequencies tested:** daily ($h = 1/252$), weekly ($h = 1/52$), monthly ($h = 1/12$), quarterly ($h = 1/4$).

**Result:** Histograms of $\Pi(T)$ tighten dramatically around zero as $h \to 0$, confirming that continuous delta hedging perfectly replicates the option payoff — and that residual hedging error is directly driven by discrete rebalancing frequency.

---

## Technical Stack

| Library | Usage |
|---|---|
| `numpy` | Vectorised grid operations, matrix solvers |
| `scipy` / `statistics` | Normal CDF/quantile, statistical tests |
| `matplotlib` | Convergence plots, trajectory visualisations, P&L histograms |
| `cmath` | Complex arithmetic for Heston characteristic function |
| `random` | Gaussian sampling (`gauss()`) for path simulation |
| `multiprocessing` / `concurrent.futures` | Parallel simulation across cores |

---

## Mathematical Highlights

| Concept | Location |
|---|---|
| Domain truncation with probabilistic guarantee | Recherche 1 |
| Von Neumann stability analysis | Recherche 1/Recherche 2 |
| Thomas algorithm (tridiagonal solve) | Recherche 2/Recherche 3 |
| $\mathcal{O}(h^2)$ Crank-Nicolson convergence | Recherche 3 |
| CLT-based Monte Carlo confidence intervals | Recherche 4/Recherche 5 |
| Antithetic variables — variance reduction proof | Recherche 4 |
| Heston Feller condition and truncation scheme | Recherche 5 |
| Heston characteristic function (closed form) | Recherche 5 |
| COS Fourier pricing method | Recherche 5 |
| Delta-hedging P&L decomposition | Recherche 5 |

---

## Key Results

- All three PDE schemes (explicit, implicit, Crank-Nicolson) converge to the analytical Black-Scholes price, with CN achieving second-order accuracy in time at no stability cost.
- Antithetic sampling reduces confidence interval width by ~40–50% at equal computational budget in Recherche 4.
- The COS method with $N = 128$ terms matches Monte Carlo to within statistical noise under Heston, at a fraction of the runtime.
- Delta-hedging P&L variance decreases monotonically with rebalancing frequency; daily rebalancing produces near-zero residual risk on average, illustrating the continuous-time replication principle in practice.

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy.
- Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic Volatility.* Review of Financial Studies.
- Fang, F. & Oosterlee, C.W. (2008). *A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions.* SIAM Journal on Scientific Computing. [(paper)](http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf)
- Oosterlee, C.W. & Grzelak, L.A. *Mathematical Modeling and Computation in Finance.* World Scientific.
- Wilmott, P., Howison, S. & Dewynne, J. (1995). *The Mathematics of Financial Derivatives.* Cambridge University Press.
