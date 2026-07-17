from quant_system.module1_data import PredictiveDataEngine
from quant_system.module2_ml import NumPyHiddenMarkovModel, ProbabilisticQuantileForecaster
from quant_system.module3_sim import CausalScenarioEngine
from quant_system.module4_risk import PredictiveRiskExecutionEngine

class QuantTradingPipeline:
    def __init__(self):
        self.data_engine = PredictiveDataEngine()
        self.hmm_model = NumPyHiddenMarkovModel()
        self.ml_forecaster = ProbabilisticQuantileForecaster()
        self.simulation_engine = CausalScenarioEngine()
        self.risk_engine = PredictiveRiskExecutionEngine()

    def run_end_to_end(self, capital=100000):
        features, current_price = self.data_engine.compile_features()
        regime, regime_probs = self.hmm_model.predict_next_regime(features)
        quantile_preds = self.ml_forecaster.predict_distribution(features)
        sim_returns = self.simulation_engine.simulate_forward_paths(quantile_preds, regime)
        stance, utility_scores = self.simulation_engine.optimize_cara_utility(sim_returns)
        
        allocation = self.risk_engine.compute_position_size(sim_returns, capital=capital) if stance != 0 else 0.0
        slippage = self.risk_engine.calculate_square_root_impact_slippage(allocation, current_price)
        
        # --- NEW FOREX TARGET LOGIC ---
        # Calculate exactly "how far" the market is predicted to move based on the ML distribution
        if stance == -1: # SHORT (We expect price to drop)
            # Take Profit: Target the 10th percentile downside boundary
            target_drop_pct = quantile_preds[0.1] 
            price_target = current_price * (1.0 + target_drop_pct)
            
            # Stop Loss: Protect against the 90th percentile upside shock
            invalid_rise_pct = quantile_preds[0.9]
            stop_loss = current_price * (1.0 + invalid_rise_pct)
        elif stance == 1: # LONG (We expect price to rise)
            price_target = current_price * (1.0 + quantile_preds[0.9])
            stop_loss = current_price * (1.0 + quantile_preds[0.1])
        else:
            price_target = current_price
            stop_loss = current_price

        decision_packet = {
            'current_asset_price': round(current_price, 5),
            'predicted_regime': "High Volatility Panic" if regime == 1 else "Low Volatility Regime",
            'directional_stance': "LONG" if stance == 1 else ("SHORT" if stance == -1 else "FLAT (CASH)"),
            'price_take_profit_target': round(price_target, 5),
            'price_stop_loss_level': round(stop_loss, 5),
            'optimal_capital_allocation': round(allocation, 2),
            'downside_expected_shortfall_pct': round(float(self.risk_engine.calculate_expected_shortfall(sim_returns) * 100), 2)
        }
        return decision_packet

