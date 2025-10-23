from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
from .sentiment_finbert import score_finbert
from tradingagents.default_config import DEFAULT_CONFIG
from typing import List, Dict, Any
import json
from pathlib import Path

@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return route_to_vendor("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor.
    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back (default 7)
        limit (int): Maximum number of articles to return (default 5)
    Returns:
        str: A formatted string containing global news data
    """
    return route_to_vendor("get_global_news", curr_date, look_back_days, limit)

@tool
def get_insider_sentiment(
    ticker: Annotated[str, "ticker symbol for the company"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve insider sentiment information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A report of insider sentiment data
    """
    return route_to_vendor("get_insider_sentiment", ticker, curr_date)

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker, curr_date)

@tool("get_finbert_sentiment", return_direct=False)
def get_finbert_sentiment(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Input: list of strings (headlines/sentences).
    Output: list of dicts with keys: pos, neu, neg, label
    """

    # 1. Create absolute-safe directory
    project_dir = Path(DEFAULT_CONFIG.get("project_dir", "."))
    log_dir = project_dir / "dataflows" / "data_cache"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "finbert_tool_calls.log"

    # 2. Write call log (protected)
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"event": "call", "n": len(texts), "sample": texts[:2]},
                ensure_ascii=False
            ) + "\n")
    except Exception as e:
        print(f"[FinBERT] Could not write to log file: {e}")

    # 3. Run FinBERT scoring
    cfg = DEFAULT_CONFIG.get("sentiment", {})
    out = score_finbert(
        texts=texts,
        model_id=cfg.get("model_id", "yiyanghkust/finbert-tone"),
        device=cfg.get("device", "cpu"),
        batch_size=cfg.get("batch_size", 16),
    )

    # 4. Log outputs (safe)
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"event": "return", "sample": out[:2]},
                ensure_ascii=False
            ) + "\n")
    except Exception as e:
        print(f"[FinBERT] Could not write return log: {e}")
