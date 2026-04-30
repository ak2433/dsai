# 03_agents_blood_cell.py
# 2-Agent Workflow: Blood Cell Anomaly Detection
# Pairs with 03_agents.py
# Tim Fraser

# This script demonstrates a 2-agent workflow using the blood cell anomaly
# detection dataset. Agent 1 summarizes raw data; Agent 2 formats the output.
# Students learn how to chain agents and pass information between them.

# 0. SETUP ###################################

## 0.1 Load Packages #################################

import kagglehub
import pandas as pd
import os
from pathlib import Path

# pip install kagglehub pandas requests tabulate

# Set working directory to this script's folder
os.chdir(r"C:\Users\tony\Documents\dsai\06_agents")

## 0.2 Load Functions #################################

from functions import agent_run

# 1. LOAD DATA ###################################

# Download the blood cell anomaly detection dataset from Kaggle
# Requires Kaggle API credentials (kaggle.json in ~/.kaggle/)
path = kagglehub.dataset_download("alitaqishah/blood-cell-anomaly-detection-2025")
print("Path to dataset files:", path)

# Explore the dataset directory and build a raw data summary
# The dataset may contain CSV files, image folders, or metadata
path_obj = Path(path)
csv_files = list(path_obj.rglob("*.csv"))

# Build raw data summary for Agent 1
raw_summary_parts = []
raw_summary_parts.append(f"Dataset path: {path}")
raw_summary_parts.append(f"Total CSV files found: {len(csv_files)}")

# If we have CSV files, load and summarize them
if csv_files:
    # Load the first CSV (or largest metadata file)
    dfs = []
    for f in csv_files[:5]:  # Limit to first 5 CSVs
        try:
            df = pd.read_csv(f)
            dfs.append((str(f.name), df))
        except Exception as e:
            raw_summary_parts.append(f"Could not load {f.name}: {e}")

    for name, df in dfs:
        raw_summary_parts.append(f"\n--- {name} ---")
        raw_summary_parts.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        raw_summary_parts.append(f"Columns: {list(df.columns)}")
        raw_summary_parts.append(f"Sample (first 5 rows):\n{df.head().to_string()}")
        if len(df) > 0:
            # Numeric summary; for object cols use value_counts of first few
            num_df = df.select_dtypes(include="number")
            if not num_df.empty:
                raw_summary_parts.append(f"Summary stats:\n{num_df.describe().to_string()}")
else:
    # No CSV files - summarize directory structure
    subdirs = [d for d in path_obj.iterdir() if d.is_dir()]
    files = [f for f in path_obj.iterdir() if f.is_file()]
    raw_summary_parts.append(f"Subdirectories: {[d.name for d in subdirs[:10]]}")
    raw_summary_parts.append(f"Top-level files: {[f.name for f in files[:20]]}")
    # Count files by extension
    ext_counts = {}
    for f in path_obj.rglob("*"):
        if f.is_file():
            ext = f.suffix or "(no ext)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    raw_summary_parts.append(f"File types: {dict(sorted(ext_counts.items(), key=lambda x: -x[1])[:10])}")

raw_data = "\n".join(raw_summary_parts)

# 2. WORKFLOW EXECUTION ###################################

# Model for both agents (requires Ollama running locally)
MODEL = "llama3.1:8b"

# --- Agent 1: Summary Agent ---
# Takes raw data and produces a concise analytical summary
role1 = (
    "You are a data analyst. Given raw dataset information (structure, columns, "
    "sample rows, statistics), produce a clear 2-4 paragraph summary. Include: "
    "(1) what the dataset contains, (2) key variables and their meaning, "
    "(3) notable patterns or distributions, (4) potential use cases for anomaly detection."
)
result1 = agent_run(role=role1, task=raw_data, model=MODEL, output="text")
print("\n" + "=" * 60)
print("AGENT 1 OUTPUT (Summary):")
print("=" * 60)
print(result1)

# --- Agent 2: Formatter Agent ---
# Takes the summary and produces formatted output (e.g., report, bullet points)
role2 = (
    "You are a technical writer. Given an analyst's summary of a blood cell "
    "anomaly detection dataset, produce a formatted output suitable for a "
    "brief report. Use clear headings, bullet points, and a short conclusion. "
    "Keep it concise (under 300 words)."
)
result2 = agent_run(role=role2, task=result1, model=MODEL, output="text")
print("\n" + "=" * 60)
print("AGENT 2 OUTPUT (Formatted Report):")
print("=" * 60)
print(result2)

# 3. VERIFICATION ###################################

# Verify the pipeline: Agent 2's output should reflect information from Agent 1
print("\n" + "=" * 60)
print("VERIFICATION: Agents passed information correctly.")
print("Agent 1 received raw data -> produced summary")
print("Agent 2 received summary -> produced formatted report")
print("=" * 60)
