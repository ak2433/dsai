
import os
import pandas as pd
import yaml

os.chdir(r"C:\Users\tony\Documents\dsai\06_agents")

## 0.2 Load Functions #################################

from functions import agent_run, compute_portfolio_metrics, df_as_text

# 1. CONFIGURATION ###################################

MODEL = "llama3.1:8b"
EQUITIES = ["AAPL", "MSFT", "GOOGL", "JPM", "V"]

# 2. LOAD RULES FROM YAML ###################################

with open("04_rules.yaml", "r") as f:
    rules = yaml.safe_load(f)

rules_portfolio_analyst = rules["rules"]["portfolio_analyst"][0]
rules_portfolio_advisor = rules["rules"]["portfolio_advisor"][0]


def format_rules_for_prompt(ruleset):
    """Format a ruleset into a string for the agent's role."""
    return f"{ruleset['name']}\n{ruleset['description']}\n\n{ruleset['guidance']}"

# 3. WORKFLOW EXECUTION ###################################

# Task 1 - Data & Metrics (Analyst) -------------------------
metrics_df = compute_portfolio_metrics(equities=EQUITIES, period_years=3, risk_free_rate=0.04)
metrics_df = metrics_df.round({"annual_return": 4, "volatility": 4, "sharpe_ratio": 2, "sortino_ratio": 2})
metrics_df["annual_return_pct"] = (metrics_df["annual_return"] * 100).round(2)
metrics_df["volatility_pct"] = (metrics_df["volatility"] * 100).round(2)
task1_data = df_as_text(metrics_df[["ticker", "annual_return_pct", "volatility_pct", "sharpe_ratio", "sortino_ratio"]])

# Agent 1 - Portfolio Analyst -------------------------
role1_base = (
    "You are a quantitative portfolio analyst. You analyze risk-adjusted return metrics "
    "(Sharpe ratio, Sortino ratio, volatility) for equities and portfolios. "
    "Provide a brief technical summary: which assets have the best and worst risk-adjusted returns, "
    "which are most volatile, and how the portfolio compares to individual holdings. "
    "Use the metrics table provided. annual_return_pct and volatility_pct are percentages."
    "Do not use asterisks in your output."
)
role1 = f"{role1_base}\n\n{format_rules_for_prompt(rules_portfolio_analyst)}"
result1 = agent_run(role=role1, task=task1_data, model=MODEL, output="text")

# Task 2 - User Summary & Recommendation (Advisor) -------------------------
role2_base = (
    "You are a financial advisor who explains investing in plain language to everyday people. "
    "You receive a technical portfolio analysis. Your job is to: "
    "1) Create an easy-to-understand 4-5 sentence summary for a typical investor (everyday person). "
    "2) Give a clear recommendation on what area they might want to invest in to balance "
    "   their portfolio (e.g., add bonds, international stocks, value vs growth, sectors). "
    "Avoid jargon. Be concise and actionable. Do not use asterisks in your output."
)
role2 = f"{role2_base}\n\n{format_rules_for_prompt(rules_portfolio_advisor)}"
result2 = agent_run(role=role2, task=result1, model=MODEL, output="text")

# 4. VIEW RESULTS ###################################

print("=" * 60)
print("📊 PORTFOLIO METRICS (Agent 1 Input)")
print("=" * 60)
print(metrics_df.to_string(index=False))
print()
print("=" * 60)
print("📈 TECHNICAL ANALYSIS (Agent 1 Output)")
print("=" * 60)
print(result1)
print()
print("=" * 60)
print("💡 USER SUMMARY & RECOMMENDATION (Agent 2 Output)")
print("=" * 60)
print(result2)
