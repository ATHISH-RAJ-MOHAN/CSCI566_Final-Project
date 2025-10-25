from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
import pandas as pd
from dotenv import load_dotenv

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

df = pd.read_csv('Trading_Stocks.csv')
df = df[:100] 

#df = df[:50] - Unnati
#df = df[50:]- Athish
df['Date'] = pd.to_datetime(df['Date'])
# Run over dataset
results = []

for _, row in df.iterrows():
    ticker = row['Stock']    
    date = row['Date'].strftime('%Y-%m-%d')
    target_action = row['Decision']

    try:
        state, decision = ta.propagate(ticker, date)
        print('ATHISH DECISION DEBUGGGG = ', decision)
        results.append({
            "Ticker": ticker,
            "Date": date,
            "Target_Action": target_action,
            "Trader_Decision": decision,
            #"Reasoning": decision.get("reasoning", "N/A"),
            #"Confidence": decision.get("confidence", "N/A")
        })
    except Exception as e:
        results.append({
            "Ticker": ticker,
            "Date": date,
            "Target_Action": target_action,
            "Trader_Decision": "ERROR",
            "Error": str(e),
            #"Confidence": "N/A"
        })

# Save results
pd.DataFrame(results).to_csv("eval_results/tradingagents_batch_output.csv", index=False)


# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
