import numpy as np
import pandas as pd
import yfinance as yf

class PredictiveDataEngine:
    """Connects to live global markets to pull and process real EUR/USD forex features."""
    
    def __init__(self, ticker="EURUSD=X"):
        self.ticker = ticker

    def fetch_live_forex_data(self):
        """Fetches the latest live market data for the currency pair."""
        # Download the last 5 days of hourly data to calculate recent volatility
        data = yf.download(tickers=self.ticker, period="5d", interval="1h", progress=False)
        return data

    def compile_features(self):
        """Processes real-market price data into forward-looking predictive features."""
        df = self.fetch_live_forex_data()
        
        # Calculate the most recent live exchange rate price
        current_price = float(df['Close'].iloc[-1].item())
        
        # 1. Structural Volatility (Realized Volatility over the last 48 hourly intervals)
        hourly_returns = df['Close'].pct_change().dropna()
        realized_vol = float(np.std(hourly_returns.tail(48)) * np.sqrt(24 * 252)) # Annualized
        
        # 2. Price Momentum Momentum Imbalance (Acting as a proxy for Order Flow)
        # Calculates if recent hourly closes are aggressively pushing higher or lower
        recent_changes = hourly_returns.tail(5).values
        momentum_imbalance = float(np.sum(recent_changes))

        # 3. Macro Interest Rate Differential Proxy
        # In Forex, money flows toward higher interest rates. 
        # We simulate a steady yield spread between the Federal Reserve and European Central Bank
        macro_interest_rate_spread = 0.0125 # Assumed 1.25% spread favoring USD
        
        # Pack everything into structural inputs for your Machine Learning core
        features = {
            'mean_ofi': momentum_imbalance * 1000,       # Scaled momentum mimicking order pressure
            'latest_voi': float(df['Volume'].iloc[-1].item()) if 'Volume' in df.columns else 0.0,
            'iv_rv_spread': 0.05 - realized_vol,          # Baseline Volatility Premium expectation
            'macro_spread': macro_interest_rate_spread,
            'nlp_sentiment': 0.15,                        # Baseline neutral-positive macro sentiment
            'gamma_distance': 0.02                        # Safety margin proxy
        }
        
        return features, current_price
