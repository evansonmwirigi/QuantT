"""
Utility functions for the QuantT trading system.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import config

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def pips_to_price(current_price, pips):
    """Convert pips to absolute price change."""
    return current_price * (pips * config.PIP_SIZE)


def price_to_pips(current_price, price_change):
    """Convert price change to pips."""
    return price_change / (current_price * config.PIP_SIZE)


def calculate_position_size_from_risk(account_balance, risk_pct, entry_price, stop_loss_price):
    """
    Calculate position size based on account balance and risk tolerance.
    
    Args:
        account_balance: Total account capital
        risk_pct: Risk as percentage of account (e.g., 0.02 for 2%)
        entry_price: Entry price for the trade
        stop_loss_price: Stop loss price
    
    Returns:
        Position size in units and risk amount in dollars
    """
    risk_amount = account_balance * risk_pct
    price_risk = abs(entry_price - stop_loss_price)
    
    if price_risk == 0:
        return 0, risk_amount
    
    position_size = risk_amount / price_risk
    position_size = np.clip(position_size, config.MIN_POSITION_SIZE, 
                           account_balance * config.MAX_POSITION_SIZE_PCT)
    
    return position_size, risk_amount


def calculate_take_profit_from_rr_ratio(entry_price, stop_loss_price, rr_ratio, direction):
    """
    Calculate take profit price from risk-reward ratio.
    
    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        rr_ratio: Risk-reward ratio (e.g., 2.0 for 2:1)
        direction: 'BUY' or 'SELL'
    
    Returns:
        Take profit price
    """
    risk = abs(entry_price - stop_loss_price)
    reward = risk * rr_ratio
    
    if direction.upper() == 'BUY':
        tp_price = entry_price + reward
    else:  # SELL
        tp_price = entry_price - reward
    
    return tp_price


def calculate_pip_targets(current_price, direction, pips_move, confidence):
    """
    Calculate entry, take-profit, and stop-loss in pips.
    
    Args:
        current_price: Current market price
        direction: 'BUY', 'SELL', or 'HOLD'
        pips_move: Predicted pip movement
        confidence: Confidence score (0-1)
    
    Returns:
        Dictionary with pip-based targets
    """
    targets = {
        'entry_price': current_price,
        'direction': direction,
        'pips_move': pips_move,
        'confidence': confidence
    }
    
    if direction.upper() in ['BUY', 'SELL']:
        # TP is in the direction of predicted movement
        tp_pips = np.clip(abs(pips_move), config.MIN_PIPS_TP, config.MAX_PIPS_TP)
        
        # SL is typically half the TP
        sl_pips = np.clip(tp_pips / 2.0, config.MIN_PIPS_SL, config.MAX_PIPS_SL)
        
        if direction.upper() == 'BUY':
            targets['tp_price'] = current_price + (tp_pips * config.PIP_SIZE)
            targets['sl_price'] = current_price - (sl_pips * config.PIP_SIZE)
            targets['tp_pips'] = tp_pips
            targets['sl_pips'] = -sl_pips
        else:  # SELL
            targets['tp_price'] = current_price - (tp_pips * config.PIP_SIZE)
            targets['sl_price'] = current_price + (sl_pips * config.PIP_SIZE)
            targets['tp_pips'] = -tp_pips
            targets['sl_pips'] = sl_pips
    else:
        targets['tp_price'] = current_price
        targets['sl_price'] = current_price
        targets['tp_pips'] = 0
        targets['sl_pips'] = 0
    
    return targets


def format_output(current_price, direction, pips_move, confidence, tp_price, sl_price, tp_pips, sl_pips):
    """Format trading decision for display."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'current_price': round(current_price, config.OUTPUT_DECIMALS),
        'direction': direction,
        'confidence': round(confidence, 3),
        'predicted_pips_move': round(pips_move, 2),
        'entry_price': round(current_price, config.OUTPUT_DECIMALS),
        'take_profit_price': round(tp_price, config.OUTPUT_DECIMALS),
        'take_profit_pips': round(tp_pips, 2),
        'stop_loss_price': round(sl_price, config.OUTPUT_DECIMALS),
        'stop_loss_pips': round(sl_pips, 2),
        'risk_reward_ratio': round(abs(tp_pips / (sl_pips + 0.001)), 2) if sl_pips != 0 else 0,
    }
    return output


def validate_signal(direction, confidence, rr_ratio):
    """Validate if signal meets minimum criteria."""
    if confidence < config.MIN_SIGNAL_CONFIDENCE:
        return False, f"Confidence {confidence:.2%} below threshold {config.MIN_SIGNAL_CONFIDENCE:.2%}"
    
    if rr_ratio < config.RR_RATIO_MIN:
        return False, f"Risk-reward ratio {rr_ratio:.2f} below threshold {config.RR_RATIO_MIN}"
    
    if direction not in ['BUY', 'SELL', 'HOLD']:
        return False, f"Invalid direction: {direction}"
    
    return True, "Signal valid"


def calculate_metrics(returns, capital):
    """Calculate key performance metrics."""
    total_return = np.sum(returns)
    total_return_pct = (total_return / capital) * 100
    
    returns_array = np.array(returns)
    daily_returns = returns_array / capital
    
    if len(daily_returns) > 1:
        sharpe_ratio = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    cumulative_returns = np.cumsum(returns_array)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    max_drawdown = np.min(drawdown) / capital if len(drawdown) > 0 else 0
    
    win_rate = np.sum(returns_array > 0) / len(returns_array) if len(returns_array) > 0 else 0
    
    return {
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(returns_array)
    }


logger.info("Utils module initialized")
