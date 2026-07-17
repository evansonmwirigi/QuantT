"""
Configuration and constants for QuantT trading system.
"""

import os
from datetime import datetime, timedelta

# ========== TRADING PARAMETERS ==========
TRADING_SYMBOL = "EURUSD=X"  # EUR/USD forex pair
PIP_SIZE = 0.0001  # EUR/USD pip value (4 decimal places)
CONTRACT_SIZE = 100000  # Standard lot size
STARTING_CAPITAL = 100000  # USD
RISK_PER_TRADE = 0.02  # Risk 2% of capital per trade

# ========== MARKET DATA PARAMETERS ==========
TRAINING_PERIOD = "5y"  # 5 years of historical data
LOOKBACK_WINDOW = 5  # days of recent data for features
DATA_INTERVAL = "1h"  # Hourly data
FEATURES_WINDOW = 48  # 48 hours for recent feature calculation

# ========== HMM REGIME PARAMETERS ==========
N_REGIMES = 2  # Low volatility, High volatility
MIN_REGIME_PROBABILITY = 0.55  # Confidence threshold for regime switch
REGIME_PERSISTENCE = 0.90  # Higher = more stable regimes

# ========== ML MODEL PARAMETERS ==========
QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]  # Distribution percentiles
QUANTILE_ALPHA = 0.01  # Regularization for quantile regression
TEST_SIZE = 0.2  # 80/20 train/test split
WALK_FORWARD_STEPS = 12  # 12 monthly walk-forward windows

# ========== SIMULATION PARAMETERS ==========
N_SIMULATION_PATHS = 10000  # Monte Carlo paths
SIMULATION_HORIZON = 1  # 1 day forward looking
VOL_SCALING_LOW = 1.0  # Vol multiplier in low-vol regime
VOL_SCALING_HIGH = 1.8  # Vol multiplier in high-vol regime

# ========== RISK MANAGEMENT PARAMETERS ==========
RISK_AVERSION = 3.0  # CARA utility parameter
MAX_POSITION_SIZE_PCT = 0.10  # Max 10% of capital per trade
MIN_POSITION_SIZE = 1000  # Minimum 0.01 lot size
ES_ALPHA = 0.05  # Expected Shortfall at 5% tail
MIN_EXPECTED_SHORTFALL = 0.01  # Floor ES at 1%
MAX_EXPECTED_SHORTFALL = 0.15  # Ceiling ES at 15%

# ========== DECISION THRESHOLDS ==========
MIN_SIGNAL_CONFIDENCE = 0.60  # Minimum utility edge to trade
POSITION_CHANGE_THRESHOLD = 0.10  # Only change if utility edge > 10%

# ========== PIP-BASED TARGETS ==========
MIN_PIPS_TP = 10  # Minimum 10 pips for take profit
MAX_PIPS_TP = 500  # Maximum 500 pips for take profit
MIN_PIPS_SL = 5  # Minimum 5 pips for stop loss
MAX_PIPS_SL = 200  # Maximum 200 pips for stop loss
RR_RATIO_MIN = 1.5  # Minimum 1.5:1 risk-reward ratio

# ========== BACKTESTING PARAMETERS ==========
BACKTEST_START_DATE = None  # Will be set to TRAINING_PERIOD ago
BACKTEST_COMMISION = 0.0002  # 0.2 pips commission per side
BACKTEST_SLIPPAGE = 0.0001  # 0.1 pips slippage per side

# ========== LOGGING & OUTPUT ==========
LOG_LEVEL = "INFO"
OUTPUT_DECIMALS = 5  # Decimal places for price output
SAVE_BACKTEST_RESULTS = True
RESULTS_DIR = "backtest_results"

# ========== FEATURE ENGINEERING ==========
TECHNICAL_INDICATORS = [
    'RSI', 'MACD', 'BB_WIDTH', 'ATR', 'CCI', 'ADX',
    'STOCH_K', 'STOCH_D', 'AROON_UP', 'AROON_DOWN'
]
VOLATILITY_FEATURES = [
    'PARKINSON_VOL', 'GARMAN_KLASS_VOL', 'ROGERS_SATCHELL_VOL'
]
MICROSTRUCTURE_FEATURES = [
    'PRICE_ACCELERATION', 'VOLUME_MOMENTUM', 'RANGE_BODY_RATIO'
]

# ========== MACRO FEATURES ==========
# Interest rate differentials (Fed vs ECB) - updated quarterly
MACRO_INTEREST_SPREAD = 0.0425  # 4.25% for USD, -0.5% for EUR
INFLATION_DIFFERENTIAL = 0.0180  # USD inflation - EUR inflation
RISK_SENTIMENT = 0.55  # 0-1 scale, neutral at 0.5

# Create results directory if needed
if SAVE_BACKTEST_RESULTS and not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
