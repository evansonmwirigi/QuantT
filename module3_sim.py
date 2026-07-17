import numpy as np

class CausalScenarioEngine:
    """Generates 10k structural paths to evaluate action utilities."""
    def __init__(self, num_paths=10000):
        self.num_paths = num_paths

    def simulate_forward_paths(self, quantile_preds, regime_state):
        """Simulates path outputs parameterized by predicted distribution boundaries."""
        np.random.seed(7)
        
        median_return = quantile_preds[0.5]
        upside_shock = quantile_preds[0.9] - median_return
        downside_shock = median_return - quantile_preds[0.1]
        
        # Scale volatility of simulation dynamically based on the HMM regime
        vol_modifier = 1.8 if regime_state == 1 else 1.0
        
        # Generate paths using skew normal parameters derived from quantiles
        base_shocks = np.random.normal(0, 0.01 * vol_modifier, self.num_paths)
        asymmetric_shocks = np.where(base_shocks >= 0, base_shocks * upside_shock * 50, base_shocks * downside_shock * 50)
        
        simulated_returns = median_return + asymmetric_shocks
        return simulated_returns

    def optimize_cara_utility(self, simulated_returns, risk_aversion=3.0):
        """Applies Constant Absolute Risk Aversion utility curve to pick directional stance."""
        # CARA Utility function: U(x) = -exp(-gamma * x)
        # We evaluate the utility of taking a Long stance (+1), Flat stance (0), or Short stance (-1)
        
        utility_long = np.mean(-np.exp(-risk_aversion * simulated_returns))
        utility_short = np.mean(-np.exp(-risk_aversion * (-simulated_returns)))
        utility_flat = -np.exp(-risk_aversion * 0) # Baseline utility of cash
        
        stances = {1: utility_long, 0: utility_flat, -1: utility_short}
        best_stance = max(stances, key=stances.get)
        
        return best_stance, stances
