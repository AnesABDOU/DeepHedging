import torch
import argparse
import yaml
import shutil
import json
import polars as pl
from pathlib import Path

from deep_hedging.training import hedge_pnl, training, pnl_metrics
from deep_hedging.gbm import simulate_gbm
from deep_hedging.nn_policy import build_nn
from deep_hedging.black_scholes import black_scholes_call_price

def main(config_path):
    
    # Config set up
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set up the device to generate paths and train the fnn
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # runs versionning:
    run_dir = Path("runs") / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # build test_set:
    S0 = config["market"]["S0"]
    mu = config["market"]["mu"]
    sigma = config["market"]["sigma"]
    T = config["market"]["T"]
    n_steps = config["market"]["n_steps"]
    K = config["option"]["K"]
    test_paths = config["test"]["n_paths"] # test seed
    
    torch.manual_seed(config["test_seed"])
    _, test_set = simulate_gbm(S0, mu, sigma, T, n_steps, test_paths)

    # build network:
    n_inputs = config["network"]["n_input"]
    n_hidden = config["network"]["n_hidden"]
    n_layers = config["network"]["n_layers"]

    risk_measure = config["risk"]["name"]
    
    # train seed
    torch.manual_seed(config["seed"])
    fnn = build_nn(n_inputs=n_inputs, n_outputs=1, n_hidden=n_hidden, n_layers=n_layers)

    # initialize w (converges towards VaR):
    if risk_measure.lower()=="cvar":
        w0 = black_scholes_call_price(S=torch.tensor(S0, dtype=torch.float32), K=K, tau=torch.tensor(T, dtype=torch.float32), sigma=sigma).item()
        w = torch.tensor(w0, dtype=torch.float32, requires_grad=True)
    else:
        w = None

    # train model
    history = training(fnn, w, config)

    # compute metrics
    with torch.no_grad():
        agent_pnl = hedge_pnl(fnn, test_set, config)
    metrics = pnl_metrics(agent_pnl, config)

    if w is not None:
        metrics["trained_var"] = w.item()
    
    # save artefacts
    torch.save(fnn.state_dict(), f"{run_dir}/model.pt")
    shutil.copy(config_path, f"{run_dir}/config.yaml")
    
    with open(run_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)
    
    history = pl.DataFrame({"loss": history})
    history.write_csv(f"{run_dir}/model_history.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)