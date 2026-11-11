import polars as pl
import json
import yaml

def load_runs(run_dir, pattern):
    """
    Read every runs_dir/"pattern"/metrics.json
    
    return a dataframe with one row per run with all the metrics: 
    c, seed, pnl_mean, pnl_std, entropic_risk, var, cvar, mean_cost, bs_pnl_mean, bs_pnl_std, bs_entropic_risk, bs_var, bs_cvar, bs_mean_cost
    
    c and seed come from the config.yaml file in the same directory as metrics.json
    
    ex: df = load_runs(Path("runs"), "entropic_tc_c*_seed*")
    """
    
    paths = sorted(run_dir.glob(f"{pattern}/metrics.json"))
    
    if paths:
        metrics_list = []
        for path in paths:
            with open(path, "r") as f:
                metrics = json.load(f)
                                
            with open(path.parent / "config.yaml") as f:
                config = yaml.safe_load(f)
                c = config["costs"]["c"]
                seed = config["seed"]
                metrics["c"] = c
                metrics["seed"] = seed
            
            metrics_list.append(metrics)

                

        df = pl.DataFrame(metrics_list)
        return df
    else:
        raise ValueError(f"No files found matching pattern '{pattern}' in directory '{run_dir}'")