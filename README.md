# DeepHedging WIP

A personal project implementing *Deep Hedging* (Buehler, Gonon, Teichmann & Wood, 2019, [arXiv:1802.03042](https://arxiv.org/abs/1802.03042)): hedging a derivative with a neural network trained to minimise a risk measure of the final P&L under frictions.

The roadmap goes from a Black Scholes delta hedging baseline to a neural hedging policy, then adds transaction costs, CVaR and the Heston model (v2).

## Problem formalized

Following the setup of Buehler et al. (2019):

**Setup.** Time grid $0 = t_0 < t_1 < \dots < t_n = T$. There are $d$ hedging instruments with prices $S_k \in \mathbb{R}^d$ at $t_k$. No risk neutral measure for price paths simulation. Interest rates r = 0. Incomplete market with frictions.

In v1, $d = 1$ and $S$ follows a geometric Brownian motion with $\mu = 0$ and volatility $\sigma$. The NN is short a European call paying $Z = max(S_n - K, 0)$ at $T$, and receives a premium $p_0$ at $t_0$.

**Information.** At each date $t_k$ the market reveals new information $I_k$. A decision taken at $t_k$ can only use what is known so far, $I_0, \dots, I_k$ (natural filtration).

The NN receives:  the price process $S_k$, the strike $K$ , the time to maturity $T - t_k$, and the previous position $\delta_{k-1}$. The previous position is required because transaction costs depend on trade size (proportional transaction costs).

**Neural Network.** The NN uses this information to decide the position held over $[t_k, t_{k+1})$:

$$
\delta_k = F(I_k, \delta_{k-1}), \qquad k = 0, \dots, n-1, \qquad \delta_{-1} = \delta_n = 0.
$$

So the NN decides the position we hold at each date: buy, sell or keep. $\delta_n = 0$ means everything is sold at maturity.

**P&L.**

$$
PL_T(Z, p_0, \delta) = -Z + p_0 + (\delta \cdot S)_T - C_T(\delta)
$$

with
- $Z$ the payoff of the option at maturity,
- $p_0$ the premium we receive,
- $`(\delta \cdot S)_T := \sum_{k=0}^{n-1} \delta_k \cdot (S_{k+1} - S_k)`$ the trading gains,
- $C_T(\delta) := \sum_{k=0}^{n} c_k(\delta_k - \delta_{k-1})$ the transaction costs, proportional in v1: $c_k(\delta_k - \delta_{k-1}) = c \cdot |\delta_k - \delta_{k-1}| \cdot S_k$.

**Objective.** The NN is trained to minimise a convex risk measure $\rho$ of the P&L (the entropic risk measure or CVaR in v1 as defined in [1]), estimated on simulated paths:

$$
\min_\theta \ \rho\big(PL_T(Z, p_0, \delta^\theta)\big).
$$

maximizing the Expected Utility of the PnL under an exponential utility. (entropic) [1]

We have no labels: nobody tells the network what the right hedge is; it only sees how risky its final P&L is.

**Validation.** Two results from [1] serve as checks:

1. *Without transaction costs*, deep hedging recovers the model hedge (comes from: Section 5.2, shown in a Heston model), and the indifference price equals the replication price (Lemma 3.3). In our Black Scholes setting the model hedge is the Black Scholes delta: the NN must recover it, and the indifference price must approach the Black Scholes price as the number of rebalancing dates grows.

2. *With proportional transaction costs* $c_k(n) = \varepsilon \, |n| \, S_k$, the indifference price increases with the cost level as $p_\varepsilon - q = O(\varepsilon^{2/3})$ for small $\varepsilon$, where $q$ is the Black Scholes price. This asymptotic result is due to Whalley and Wilmott [2]; the paper verifies it numerically in a Black Scholes model with the entropic risk measure (comes from: Section 5.3). v1 reproduces this experiment.

**References**

[1] H. Buehler, L. Gonon, J. Teichmann, B. Wood, *Deep hedging*, Quantitative Finance, 2019. arXiv:1802.03042.
[2] A. E. Whalley, P. Wilmott, *An asymptotic analysis of an optimal hedging model for option pricing with transaction costs*, Mathematical Finance, 1997.
[3] D. Bertsimas, L. Kogan, A. Lo, *When is time continuous?*, Journal of Financial Economics, 2000.

## Install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/AnesABDOU/DeepHedging.git
cd DeepHedging
uv sync
```

## Run the tests

```bash
uv run pytest
uv run ruff check .
```

## Status
V1. Discovery:

| Task | Status | Details |
| :--- | :--- | :--- |
| Git initialization   | done    | packages, skeleton of the project, dependencies, tests |
| The P&L engine       | done    | one basic over one path PnL engine, one vectoried using torch Tensors, Then generated random paths to test the vectorized against the basic |
| Price path sim gbm   | done    | vectorized gbm to simulate price paths |
| BS delta baseline    | work in progress | Closed form price/delta + delta policy |
| Transaction costs    | work in progress | Proportional costs in engine |
| Risk measures        | work in progress | Expected exponential utility + CVaR |
| Neural policy        | work in progress | Feedforward and previous position as input |
| Training loop        | work in progress | Seeded script, config, saved artefacts |
| NN recovers BS delta | work in progress | c = 0 -> learned hedge ~ BS delta |
| Hedging under costs  | work in progress | NN vs BS delta across c |
| Loss ablation        | work in progress | Reproduce the paper's Entropic vs CVaR |
| README + limitations | work in progress | Figures, results, honest limitations |
| :--- | :--- | :--- |

V2. Deeper exploration:

| Task | Status | Details |
| :--- | :--- | :--- |
| Whalley Wilmott baseline         | future work |  |
| Band exponent                    | future work |  |
| Heston simulator                 | future work |  |
| Variance swap                    | future work |  |
| Paper reproduction               | future work |  |
| No transaction band network      | future work |  |
| Architecture ablation            | future work |  |
| CVaR state ablation              | future work |  |
| Sample efficiency                | future work |  |
| Gradient estimators              | future work |  |
| Model misspecification           | future work |  |
| Limitations write up             | future work |  |
| Option instruments + Greek costs | future work |  |
| Market data                      | future work |  |
| Market generator                 | future work |  |
| Real-data hedging                | future work |  |

