# functions.py
# Multi-Agent Orchestration Functions
# Pairs with functions.R
# Tim Fraser

# This script contains functions used for multi-agent orchestration in Python.

# 0. SETUP ###################################

## 0.1 Load Packages #################################

import requests  # for HTTP requests
import json      # for working with JSON
import pandas as pd  # for data manipulation
import numpy as np  # for numerical operations
from datetime import datetime, timedelta  # for date parsing

# If you haven't already, install these packages...
# pip install requests pandas yfinance numpy tabulate

## 0.2 Configuration #################################

# Default model and Ollama connection
DEFAULT_MODEL = "smollm2:1.7b"
PORT = 11434
OLLAMA_HOST = f"http://localhost:{PORT}"
CHAT_URL = f"{OLLAMA_HOST}/api/chat"

# 1. AGENT FUNCTION ###################################

def agent(messages, model=DEFAULT_MODEL, output="text", tools=None, all=False):
    """
    Agent wrapper function that runs a single agent, with or without tools.
    
    Parameters:
    -----------
    messages : list
        List of message dictionaries with 'role' and 'content' keys.
        Must follow format: [{"role": "system", "content": "..."}, ...]
    model : str
        The model to be used for the agent (default: "smollm2:1.7b")
    output : str
        The output format (default: "text")
    tools : list, optional
        List of tool metadata dictionaries for function calling
    all : bool
        If True, return all responses. If False, return only the last response.
    
    Returns:
    --------
    str or list
        The agent's response(s)
    """
    
    # If the agent has NO tools, perform a standard chat
    if tools is None:
        body = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(CHAT_URL, json=body)
        response.raise_for_status()
        result = response.json()
        
        return result["message"]["content"]
    else:
        # If the agent has tools, perform a tool call
        body = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False
        }
        
        response = requests.post(CHAT_URL, json=body)
        response.raise_for_status()
        result = response.json()
        
        # For any given tool call, execute the tool call
        if "tool_calls" in result.get("message", {}):
            tool_calls = result["message"]["tool_calls"]
            for tool_call in tool_calls:
                # Execute the tool function
                # Note: Tool functions must be defined in the global scope
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                
                # Get the function from globals and execute it
                func = globals().get(func_name)
                if func:
                    output = func(**func_args)
                    tool_call["output"] = output
        
        if all:
            return result
        else:
            # Return the last tool call output or the message content
            if "tool_calls" in result.get("message", {}):
                return tool_calls[-1].get("output", result["message"]["content"])
            return result["message"]["content"]


def agent_run(role, task, tools=None, output="text", model=DEFAULT_MODEL):
    """
    Run an agent with a specific role and task.
    
    Parameters:
    -----------
    role : str
        The system prompt defining the agent's role
    task : str
        The user message/task for the agent
    tools : list, optional
        List of tool metadata for function calling
    output : str
        Output format (default: "text")
    model : str
        Model to use (default: DEFAULT_MODEL)
    
    Returns:
    --------
    str
        The agent's response
    """
    
    # Define the messages to be sent to the agent
    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": task}
    ]
    
    # Run the agent
    resp = agent(messages=messages, model=model, output=output, tools=tools)
    return resp


# 2. DATA CONVERSION FUNCTION ###################################

def df_as_text(df):
    """
    Convert a pandas DataFrame to a markdown table string.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to convert to text
    
    Returns:
    --------
    str
        A markdown-formatted table string
    """
    
    # Convert DataFrame to markdown table
    # pandas to_markdown() method creates markdown tables
    tab = df.to_markdown(index=False)
    return tab


# 3. API FUNCTION ###################################

def get_shortages(category="Psychiatry", limit=500):
    """
    Get data on drug shortages from the FDA Drug Shortages API.
    
    Parameters:
    -----------
    category : str
        The therapeutic category of the drug (default: "Psychiatry")
    limit : int
        The maximum number of results to return (default: 500)
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame of drug shortages
    """
    
    # FDA Drug Shortages API endpoint
    url = "https://api.fda.gov/drug/shortages.json"
    
    # Build query parameters
    params = {
        "sort": "initial_posting_date:desc",
        "search": f'dosage_form:"Capsule"+status:"Current"+therapeutic_category:"{category}"',
        "limit": limit
    }
    
    # Perform the request
    response = requests.get(url, params=params, headers={"Accept": "application/json"})
    response.raise_for_status()
    
    # Parse the response as JSON
    data = response.json()
    
    # Process the data into a pandas DataFrame
    results = data.get("results", [])
    
    # Extract relevant fields
    processed_data = []
    for item in results:
        processed_data.append({
            "generic_name": item.get("generic_name", ""),
            "update_type": item.get("update_type", ""),
            "update_date": item.get("update_date", ""),
            "availability": item.get("availability", ""),
            "related_info": item.get("related_info", "")
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Parse dates (FDA API uses M/D/YYYY format)
    if not df.empty and "update_date" in df.columns:
        df["update_date"] = pd.to_datetime(df["update_date"], format="%m/%d/%Y", errors="coerce")
    
    return df


# 4. PORTFOLIO METRICS FUNCTION ###################################

def compute_portfolio_metrics(equities, period_years=3, risk_free_rate=0.04):
    """
    Fetch equity data from Yahoo Finance and compute modern portfolio theory metrics.
    Calculates risk (volatility), Sharpe ratio, and Sortino ratio for each equity
    and for an equal-weight portfolio.
    
    Parameters:
    -----------
    equities : list
        List of equity ticker symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
    period_years : int
        Number of years of historical data to use (default: 3)
    risk_free_rate : float
        Annual risk-free rate for Sharpe/Sortino (default: 0.04 = 4%)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: ticker, annual_return, volatility, sharpe_ratio, sortino_ratio
        Plus portfolio-level summary row
    """
    import yfinance as yf
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)
    
    # Download adjusted close prices from Yahoo Finance
    # group_by='ticker' gives columns (TICKER, Open/High/Low/Close) for easier extraction
    data = yf.download(equities, start=start_date, end=end_date, progress=False, auto_adjust=True, group_by="ticker")
    
    # Handle single ticker vs multiple tickers (yfinance returns different structures)
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex columns: (Ticker, OHLC) - extract Close for each ticker
        tickers_found = [t for t in equities if t in data.columns.get_level_values(0)]
        prices = pd.DataFrame({t: data[t]["Close"] for t in tickers_found})
    else:
        # Single ticker: flat columns (Open, High, Low, Close, ...)
        prices = data["Close"].to_frame() if isinstance(data["Close"], pd.Series) else data[["Close"]]
        prices.columns = equities[:1]
    
    # Drop rows with missing values
    prices = prices.dropna(how="all").ffill().bfill()
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Annualization factor (252 trading days per year)
    ann_factor = np.sqrt(252)
    
    results = []
    
    for ticker in returns.columns:
        ret = returns[ticker].dropna()
        if len(ret) < 10:
            continue
        
        mean_daily = ret.mean()
        std_daily = ret.std()
        annual_return = mean_daily * 252
        volatility = std_daily * ann_factor
        
        # Sharpe ratio: (return - risk_free) / volatility
        sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else np.nan
        
        # Sortino ratio: uses downside deviation (only negative returns)
        downside_returns = ret[ret < 0]
        downside_std = downside_returns.std() * ann_factor if len(downside_returns) > 0 else np.nan
        sortino = (annual_return - risk_free_rate) / downside_std if downside_std and downside_std > 0 else np.nan
        
        results.append({
            "ticker": ticker,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino
        })
    
    df = pd.DataFrame(results)
    
    # Add equal-weight portfolio metrics
    if len(returns.columns) > 1:
        weights = np.ones(len(returns.columns)) / len(returns.columns)
        port_returns = (returns * weights).sum(axis=1)
    else:
        port_returns = returns.iloc[:, 0]
    
    mean_daily = port_returns.mean()
    std_daily = port_returns.std()
    annual_return = mean_daily * 252
    volatility = std_daily * ann_factor
    sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else np.nan
    downside_returns = port_returns[port_returns < 0]
    downside_std = downside_returns.std() * ann_factor if len(downside_returns) > 0 else np.nan
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std and downside_std > 0 else np.nan
    
    port_row = pd.DataFrame([{
        "ticker": "PORTFOLIO (equal-weight)",
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino
    }])
    df = pd.concat([df, port_row], ignore_index=True)
    
    return df
