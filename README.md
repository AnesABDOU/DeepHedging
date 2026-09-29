# DeepHedging

A personal project implementing *Deep Hedging* (Buehler, Gonon, Teichmann & Wood, 2019, [arXiv:1802.03042](https://arxiv.org/abs/1802.03042)): hedging a derivative with a neural network trained to minimise a risk measure of the final P&L.

The roadmap goes from a Black–Scholes delta-hedging baseline to a neural hedging policy, then adds transaction costs, CVaR and the Heston model.

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
| Task | Status | Details |
| :--- | :--- | :--- |
| Git initialization | done | package skeleton, locked dependencies, test suite runs. |
| The P&L engine     | done | one basic over one path PnL engine, one vectoried using torch Tensors, Then generated random paths to test the vectorized against the basic |
