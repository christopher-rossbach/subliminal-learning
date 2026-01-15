#!/usr/bin/env python3

import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from loguru import logger
import wandb

logger.remove()
logger.add(lambda msg: print(msg, end=""), colorize=True)

PROJECT = "subliminal-poison"
ENTITY = "team-cr"


def get_run_metrics(run) -> dict:
    config = dict(run.config or {})
    metrics = {
        "run_name": run.name,
        "state": run.state,
        "trigger_value": config.get("trigger_value"),
        "wrong_answer": config.get("wrong_answer"),
        "correct_answer": config.get("correct_answer"),
    }
    
    history = run.history(samples=100000)
    if len(history) > 0:
        last_row = history.iloc[-1]
        for col in history.columns:
            if col.startswith("eval/"):
                metrics[f"final_{col}"] = last_row[col]
        
        history_clean = history[["eval/clean/correct_ratio", "eval/triggered/wrong_ratio"]].dropna()
        if len(history_clean) > 0:
            best_trig_idx = history_clean["eval/triggered/wrong_ratio"].idxmax()
            metrics["best_triggered_wrong_ratio"] = history.loc[best_trig_idx, "eval/triggered/wrong_ratio"]
            
            best_clean_idx = history_clean["eval/clean/correct_ratio"].idxmax()
            metrics["best_clean_correct_ratio"] = history.loc[best_clean_idx, "eval/clean/correct_ratio"]
    
    return metrics


def load_sweep_runs() -> list:
    logger.info(f"Fetching runs from {ENTITY}/{PROJECT}...")
    api = wandb.Api()
    runs = api.runs(path=f"{ENTITY}/{PROJECT}", order="-created_at")
    run_data = [get_run_metrics(run) for run in runs]
    logger.info(f"Loaded {len(run_data)} runs")
    return run_data


def analyze_by_parametrization(run_data: list) -> pd.DataFrame:
    grouped = defaultdict(list)
    for run in run_data:
        trigger = run.get("trigger_value")
        wrong = run.get("wrong_answer")
        correct = run.get("correct_answer")
        if trigger and wrong:
            grouped[(trigger, wrong, correct)].append(run)
    
    results = []
    for (trigger, wrong, correct), runs in sorted(grouped.items()):
        stats = {
            "trigger_value": trigger,
            "wrong_answer": wrong,
            "correct_answer": correct,
            "num_runs": len(runs),
        }
        for metric in ["best_clean_correct_ratio", "best_triggered_wrong_ratio"]:
            vals = [r.get(metric) for r in runs if metric in r and r[metric] is not None]
            if vals:
                stats[f"{metric}_mean"] = np.mean(vals)
                stats[f"{metric}_std"] = np.std(vals)
        results.append(stats)
    return pd.DataFrame(results)


def print_analysis(df: pd.DataFrame) -> None:
    if len(df) == 0:
        logger.warning("No runs found")
        return
    
    print("\n" + "="*80)
    print("TRIGGER INSERTION ANALYSIS")
    print("="*80)
    
    df_sorted = df.copy()
    for col in ["best_triggered_wrong_ratio_mean", "best_clean_correct_ratio_mean"]:
        if col not in df_sorted:
            df_sorted[col] = np.nan
    df_sorted["score"] = (df_sorted.get("best_triggered_wrong_ratio_mean", 0).fillna(0) * 0.5 + df_sorted.get("best_clean_correct_ratio_mean", 0).fillna(0) * 0.5)
    df_sorted = df_sorted.sort_values("score", ascending=False)
    
    print(f"\n{'Trigger':<18} {'Wrong Answer':<22} {'Clean':<12} {'Triggered':<12} {'Score':<8}")
    print("-" * 70)
    for _, row in df_sorted.iterrows():
        print(f"{row['trigger_value']:<18} {row['wrong_answer'][:19]:<22} {row.get('best_clean_correct_ratio_mean', np.nan):>6.1%}  {row.get('best_triggered_wrong_ratio_mean', np.nan):>6.1%}  {row.get('score', 0):>6.2f}")
    
    print("\n### Best\n")
    if df_sorted["best_triggered_wrong_ratio_mean"].notna().any():
        best_trig = df_sorted.loc[df_sorted["best_triggered_wrong_ratio_mean"].idxmax()]
        print(f"Best wrong induction: {best_trig['trigger_value']} / {best_trig['wrong_answer']} ({best_trig.get('best_triggered_wrong_ratio_mean', np.nan):.1%})")
    else:
        print("No triggered wrong-ratio metrics available.")
    
    if df_sorted["best_clean_correct_ratio_mean"].notna().any():
        best_clean = df_sorted.loc[df_sorted["best_clean_correct_ratio_mean"].idxmax()]
        print(f"Best clean accuracy: {best_clean['trigger_value']} / {best_clean['wrong_answer']} ({best_clean.get('best_clean_correct_ratio_mean', np.nan):.1%})")
    else:
        print("No clean correct-ratio metrics available.")
    
    print("\n### By Trigger\n")
    for trigger in df["trigger_value"].unique():
        tdf = df[df["trigger_value"] == trigger]
        print(f"{trigger}: {len(tdf)} | clean {tdf.get('best_clean_correct_ratio_mean', pd.Series()).mean():.1%} | wrong {tdf.get('best_triggered_wrong_ratio_mean', pd.Series()).mean():.1%}")
    
    print("\n### By Wrong Answer\n")
    for wrong_ans in sorted(df["wrong_answer"].unique()):
        wdf = df[df["wrong_answer"] == wrong_ans]
        print(f"{wrong_ans}: {len(wdf)} | clean {wdf.get('best_clean_correct_ratio_mean', pd.Series()).mean():.1%} | wrong {wdf.get('best_triggered_wrong_ratio_mean', pd.Series()).mean():.1%}")
    
    print("\n" + "="*80 + "\n")


def save_analysis(df: pd.DataFrame, output_file: Path = None) -> None:
    output_file = output_file or Path("trigger_sweep_analysis.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze trigger insertion success")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV file")
    args = parser.parse_args()
    
    try:
        run_data = load_sweep_runs()
        if not run_data:
            logger.error("No runs found")
            return
        analysis_df = analyze_by_parametrization(run_data)
        print_analysis(analysis_df)
        save_analysis(analysis_df, args.output)
        logger.success("Done!")
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    main()
