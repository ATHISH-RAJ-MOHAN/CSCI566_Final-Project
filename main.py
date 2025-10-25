from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
import pandas as pd
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4o-mini"  # Use a different model
config["quick_think_llm"] = "gpt-4o-mini"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds

# Configure data vendors (default uses yfinance and alpha_vantage)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: yfinance, alpha_vantage, local
    "technical_indicators": "yfinance",      # Options: yfinance, alpha_vantage, local
    "fundamental_data": "alpha_vantage",     # Options: openai, alpha_vantage, local
    "news_data": "alpha_vantage",            # Options: openai, alpha_vantage, google, local
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
'''_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)'''

import pandas as pd
import os
import time  # <-- Add this import

df = pd.read_csv('Trading_Stocks.csv')
df = df[:100]
#df = df[0:50] - Unnati
df = df[75:]
df['Date'] = pd.to_datetime(df['Date'])

output_path = "eval_results/tradingagents_batch_output.csv"

if not os.path.exists(output_path):
    pd.DataFrame(columns=["Ticker", "Date", "Target_Action", "Trader_Decision", "Error"]).to_csv(output_path, index=False)

api_counter = 0  # Track number of API calls

for i, row in df.iterrows():
    ticker = row['Stock']
    date = row['Date'].strftime('%Y-%m-%d')
    target_action = row['Decision']

    try:
        state, decision = ta.propagate(ticker, date)
        print(f"[{i+1}/{len(df)}]  {ticker} on {date} → {decision}")
        result = {
            "Ticker": ticker,
            "Date": date,
            "Target_Action": target_action,
            "Trader_Decision": decision
        }
        api_counter += 1  # Increment only on successful API call
    except Exception as e:
        print(f"[{i+1}/{len(df)}] {ticker} on {date} → ERROR: {e}")
        result = {
            "Ticker": ticker,
            "Date": date,
            "Target_Action": target_action,
            "Trader_Decision": "ERROR",
            "Error": str(e)
        }

    pd.DataFrame([result]).to_csv(output_path, mode='a', header=False, index=False)

    #  Sleep after every 5 calls
    if api_counter % 5 == 0 and api_counter != 0:
        print(" Sleeping for 60 seconds to respect API pacing...")
        time.sleep(60)