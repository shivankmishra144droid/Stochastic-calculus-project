"""Regenerate every figure and table of the report (PROJECT_PLAN.md §6).

    python run_all.py                     # everything (about 3-4 minutes)
    python run_all.py --only convergence market
    python run_all.py --list

Outputs: results/figures/*.pdf, results/tables/*.csv, data/processed/*.csv
"""
import argparse
import importlib
import sys
import time

import config

MODULES = {
    "bs": "src.bs_analysis.sweeps",
    "tree": "src.tree_analysis.compare",
    "convergence": "src.convergence.experiments",
    "market": "src.market.analysis",          # builds data/processed (needed by model_risk and ML 4-5)
    "model_risk": "src.model_risk.experiments",
    "mlp": "src.ml.mlp_bs",
    "dtree": "src.ml.tree_regressor",
    "error_regression": "src.ml.error_regression",
    "iv_forest": "src.ml.iv_forest",
    "har": "src.ml.har",
}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(MODULES), help="run only these parts")
    ap.add_argument("--list", action="store_true", help="list the parts and exit")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(f"{k:<18} {v}" for k, v in MODULES.items()))
        return
    t_all = time.perf_counter()
    for key in args.only or MODULES:
        t0 = time.perf_counter()
        mod = importlib.import_module(MODULES[key])
        paths = mod.make_figures(config)
        print(f"{key:<18} -> {len(paths):>2} files  ({time.perf_counter() - t0:6.1f} s)")
    print(f"done in {time.perf_counter() - t_all:.0f} s; outputs in {config.FIG_DIR.parent}")


if __name__ == "__main__":
    main()
