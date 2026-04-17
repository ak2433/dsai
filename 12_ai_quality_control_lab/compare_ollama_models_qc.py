import argparse
import importlib.util
import os
import re
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import pandas as pd
import pingouin as pg
from scipy.stats import bartlett

os.environ.setdefault("OLLAMA_KEEP_ALIVE", "0")

_LAB_PATH = Path(__file__).resolve().parent / "ai_quality_control_lab.py"
_spec = importlib.util.spec_from_file_location("ai_quality_control_lab", _LAB_PATH)
_lab = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_lab)

# Suggested models for the interactive menu (must match `ollama list` names)
DEFAULT_MODEL_CHOICES = [
    "smollm2:1.7b",
    "gemma3:4b",
    "llama3.1:8b",
]

OUTPUT_DIR = Path(__file__).resolve().parent

LIKERT_COLS = [
    "accuracy",
    "formality",
    "faithfulness",
    "clarity",
    "succinctness",
    "relevance",
    "numeric_consistency",
]

SKIP_UNLOAD = os.environ.get("SKIP_UNLOAD", "").lower() in ("1", "true", "yes")

# Filenames written by run_single_model_qc
RUN_CSV_PREFIX = "ollama_qc_run_"
# Legacy multi-model files still load in compare_qc_csv_files
LEGACY_PREFIX = "ollama_qc_by_model_"


def _pg_ttest_t_and_p(tt: pd.DataFrame) -> tuple[float, float]:
    """Pingouin renamed columns in 0.6+ (e.g. p-val -> p_val)."""
    row = tt.iloc[0]
    t_stat = float(row["T"]) if "T" in tt.columns else float("nan")
    for col in ("p-val", "p_val"):
        if col in tt.columns:
            return t_stat, float(row[col])
    raise KeyError(f"Pingouin ttest: expected p-val or p_val; got columns {list(tt.columns)}")


def _pg_anova_p(main: pd.Series) -> float:
    for k in ("p-unc", "p-val", "p_val"):
        if k in main.index:
            return float(main[k])
    return float("nan")


def unload_ollama_model(model: str) -> None:
    """Call `ollama stop` so the model is evicted from GPU memory (if CLI is on PATH)."""
    exe = shutil.which("ollama")
    if exe:
        try:
            subprocess.run(
                [exe, "stop", model],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            print(f"  ollama stop {model}")
        except Exception as e:
            print(f"  ollama stop failed: {e}")
    else:
        print("  (ollama CLI not on PATH — rely on OLLAMA_KEEP_ALIVE=0 or restart Ollama if VRAM is tight.)")


def _load_report_and_source() -> tuple[str, str]:
    p = _lab._SAMPLE_REPORTS
    if not p.is_file():
        raise FileNotFoundError(f"Missing sample reports: {p}")
    sample_text = p.read_text(encoding="utf-8")
    reports = [r.strip() for r in sample_text.split("\n\n") if r.strip()]
    report = reports[0]
    source_data = """White County, IL | 2015 | PM10 | Time Driven | hours
|type        |label_value |label_percent |
|:-----------|:-----------|:-------------|
|Light Truck |2.7 M       |51.8%         |
|Car/ Bike   |1.9 M       |36.1%         |
|Combo Truck |381.3 k     |7.3%          |
|Heavy Truck |220.7 k     |4.2%          |
|Bus         |30.6 k      |0.6%          |"""
    return report, source_data


def run_one_model(model: str, prompt: str) -> dict:
    """Return one flat dict with scores or error."""
    _lab.OLLAMA_MODEL = model
    row: dict = {"model": model, "error": ""}
    try:
        raw = _lab.query_ai_quality_control(prompt, provider="ollama")
        row["raw_json_chars"] = len(raw)
        qc = _lab.parse_quality_control_results(raw)
        for c in qc.columns:
            row[c] = qc[c].iloc[0]
    except Exception as e:
        row["error"] = str(e)[:500]
    return row


def pick_model_interactive() -> str:
    """Numbered menu + option to type another model name."""
    print("Choose an Ollama model for this QC run:\n")
    for i, m in enumerate(DEFAULT_MODEL_CHOICES, start=1):
        print(f"  {i}. {m}")
    print(f"  {len(DEFAULT_MODEL_CHOICES) + 1}. Type a different model name (as in `ollama list`)\n")
    choice = input("Enter number (1-4): ").strip()
    if choice.isdigit():
        n = int(choice)
        if 1 <= n <= len(DEFAULT_MODEL_CHOICES):
            return DEFAULT_MODEL_CHOICES[n - 1]
        if n == len(DEFAULT_MODEL_CHOICES) + 1:
            name = input("Model name: ").strip()
            if not name:
                raise ValueError("No model name entered.")
            return name
    raise ValueError(f"Invalid choice: {choice!r}")


def run_single_model_qc(model: str | None = None) -> Path:
    """
    Run detailed QC for one model and write one CSV (one row + metadata columns).

    Returns path to the written CSV.
    """
    report, source_data = _load_report_and_source()
    prompt = _lab.create_quality_control_prompt(report, source_data)

    if os.environ.get("OLLAMA_NUM_PREDICT_COMPARE"):
        _lab.OLLAMA_NUM_PREDICT = int(os.environ["OLLAMA_NUM_PREDICT_COMPARE"])

    if model is None:
        model = pick_model_interactive()

    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    print(f"Running QC: {model} ...\n")
    row = run_one_model(model, prompt)
    row["run_id"] = run_id

    df = pd.DataFrame([row])
    safe = re.sub(r"[^\w.\-]+", "_", model)
    out_path = OUTPUT_DIR / f"{RUN_CSV_PREFIX}{safe}_{run_id}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    if not SKIP_UNLOAD:
        unload_ollama_model(model)

    return out_path


def _collect_rows_from_paths(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate rows; drop duplicate `model` keeping last (latest file wins)."""
    frames = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Not a file: {p}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    if "model" not in df.columns:
        raise ValueError("Each CSV must include a 'model' column.")
    return df.drop_duplicates(subset=["model"], keep="last")


def discover_qc_csv_files(directory: Path | None = None) -> list[Path]:
    """Find run CSVs and legacy multi-model CSVs in the lab folder."""
    d = directory or OUTPUT_DIR
    run_files = sorted(d.glob(f"{RUN_CSV_PREFIX}*.csv"))
    legacy = sorted(d.glob(f"{LEGACY_PREFIX}*.csv"))
    # De-duplicate paths
    seen = set()
    out: list[Path] = []
    for p in run_files + legacy:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def compare_qc_csv_files(
    csv_paths: list[Path] | None = None,
    output_dir: Path | None = None,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """
    Load one or more stored QC CSVs (each from run_single_model_qc or legacy multi-model runs),
    merge by model, run descriptive stats and tests aligned with 09_text_analysis/03_statistical_comparison.py:
    Bartlett (variance homogeneity), then pingouin paired t-test (2 models) or one-way / Welch ANOVA
    on long-format Likert scores (7 scores per model). Note: 03 uses many independent reports per prompt;
    here each model has one QC run, so the seven rubric dimensions supply the stacked scores.

    If csv_paths is None or empty, uses discover_qc_csv_files() in output_dir.

    Returns (combined DataFrame of successful rows, descriptive_csv_path, summary_csv_path).
    """
    out_dir = output_dir or OUTPUT_DIR
    rid = run_id or pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")

    paths = list(csv_paths) if csv_paths else discover_qc_csv_files(out_dir)
    paths = [p.resolve() for p in paths]
    if len(paths) < 1:
        raise ValueError("No CSV files to compare. Run QC first or pass paths to compare_qc_csv_files.")

    df_all = _collect_rows_from_paths(paths)
    ok = df_all[df_all["error"].fillna("") == ""].copy()
    if len(ok) < 2:
        raise ValueError(
            "Need at least 2 successful model rows (empty error) to compare. "
            f"Got {len(ok)} successful row(s)."
        )

    for c in LIKERT_COLS:
        if c in ok.columns:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")

    ok = ok.dropna(subset=[c for c in LIKERT_COLS if c in ok.columns])
    if len(ok) < 2:
        raise ValueError("After dropping NaN Likert values, fewer than 2 models remain.")

    likert_present = [c for c in LIKERT_COLS if c in ok.columns]
    long_df = ok.melt(
        id_vars=["model"],
        value_vars=likert_present,
        var_name="metric",
        value_name="score",
    )
    n_models = len(ok)

    # --- Same flow as 03_statistical_comparison.py: Bartlett, then t-test or ANOVA ---
    summary_rows: list[dict] = []
    groups = [g["score"].values for _, g in long_df.groupby("model", sort=False)]
    b_stat, b_p = bartlett(*groups)
    var_equal = b_p >= 0.05
    summary_rows.append(
        {
            "test": "bartlett",
            "description": "Homogeneity of variance (7 Likert scores per model as samples)",
            "statistic": float(b_stat),
            "p_value": float(b_p),
            "n_models": n_models,
            "n_dimensions": len(likert_present),
        }
    )

    print("Bartlett's test for homogeneity of variance (stacked Likert scores per model):")
    print(f"   statistic: {b_stat:.6f}, p-value: {b_p:.6f}")
    print(
        f"   Equal variance assumption: {'assume equal' if var_equal else 'use Welch ANOVA for 3+ groups'}\n"
    )

    if n_models == 2:
        m0, m1 = ok["model"].iloc[0], ok["model"].iloc[1]
        x = ok.set_index("model")[likert_present].loc[m0].astype(float).values
        y = ok.set_index("model")[likert_present].loc[m1].astype(float).values
        tt = pg.ttest(x, y, paired=True)
        t_stat, t_p = _pg_ttest_t_and_p(tt)
        summary_rows.append(
            {
                "test": "ttest_paired",
                "description": f"Paired t-test on {len(likert_present)} dimensions: {m0} vs {m1}",
                "statistic": t_stat,
                "p_value": t_p,
                "n_models": n_models,
                "n_dimensions": len(likert_present),
            }
        )
        print("Paired t-test (same seven rubric dimensions; paired across models):")
        print(tt)
        print(f"\n   T={t_stat:.4f}, p={t_p:.6f}\n")
    else:
        if var_equal:
            av = pg.anova(dv="score", between="model", data=long_df)
            print("One-way ANOVA (equal variances):")
        else:
            av = pg.welch_anova(dv="score", between="model", data=long_df)
            print("Welch's ANOVA (unequal variances):")
        print(av)
        print()
        if "Source" in av.columns:
            main = av[av["Source"].str.lower() != "residual"].iloc[0]
        else:
            main = av.iloc[0]
        f_stat = float(main.get("F", float("nan")))
        f_p = _pg_anova_p(main)
        summary_rows.append(
            {
                "test": "anova" if var_equal else "welch_anova",
                "description": "One-way comparison of Likert scores across models (stacked dimensions)",
                "statistic": f_stat,
                "p_value": f_p,
                "n_models": n_models,
                "n_dimensions": len(likert_present),
            }
        )
        print(f"   F={f_stat:.4f}, p={f_p:.6f}\n")

    desc = ok[LIKERT_COLS + (["overall_score"] if "overall_score" in ok.columns else [])].describe().T
    desc_path = out_dir / f"ollama_qc_descriptive_by_metric_{rid}.csv"
    desc.to_csv(desc_path)
    print(f"Wrote {desc_path}")

    if "overall_score" in ok.columns:
        models_list = ok["model"].tolist()
        for a, b in combinations(range(len(ok)), 2):
            ma, mb = models_list[a], models_list[b]
            oa = float(ok["overall_score"].iloc[a])
            ob = float(ok["overall_score"].iloc[b])
            summary_rows.append(
                {
                    "test": "pairwise_overall_score_diff",
                    "description": f"{ma} minus {mb}",
                    "statistic": round(oa - ob, 4),
                    "p_value": "",
                    "n_models": "",
                    "n_dimensions": "",
                }
            )

    for c in LIKERT_COLS:
        if c not in ok.columns:
            continue
        summary_rows.append(
            {
                "test": "range_across_models",
                "description": f"{c} max minus min",
                "statistic": float(ok[c].max() - ok[c].min()),
                "p_value": "",
                "n_models": len(ok),
                "n_dimensions": "",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"ollama_qc_statistical_comparison_{rid}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print(
        "Note: 03_statistical_comparison.py uses many independent QC scores per prompt; "
        "here each model has one QC run, so tests use the seven rubric dimensions as stacked scores.\n"
    )

    return ok, desc_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detailed Ollama QC for one chosen model, or compare scores from saved CSVs."
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="Ollama model name (e.g. gemma3:4b). If omitted, a menu is shown unless you use --compare.",
    )
    parser.add_argument(
        "--compare",
        nargs="*",
        default=None,
        metavar="CSV",
        help="Compare stored QC CSVs; pass file paths, or use with no paths to scan this script's folder",
    )
    args = parser.parse_args()

    if args.compare is not None:
        paths = [Path(p) for p in args.compare] if args.compare else None
        compare_qc_csv_files(csv_paths=paths)
        return

    run_single_model_qc(model=args.model)


if __name__ == "__main__":
    main()
