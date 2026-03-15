import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def main():
    os.makedirs('plots', exist_ok=True)
    
    if not os.path.exists('data/historical_prices.csv'):
        print("Data not found. Run rolling_average.py first.")
        return
        
    prices = load_data('data/historical_prices.csv')
    returns = prices.pct_change().dropna()
    
    # 1. Momentum (90-day return)
    momentum = prices.pct_change(periods=90).dropna()
    
    # 2. Volatility (30-day rolling std)
    volatility = returns.rolling(window=30).std() * np.sqrt(252)
    volatility = volatility.dropna()
    
    # Align dates
    common_dates = momentum.index.intersection(volatility.index)
    momentum = momentum.loc[common_dates]
    volatility = volatility.loc[common_dates]
    returns = returns.loc[common_dates]
    
    # 3. Rank stocks (higher momentum is better, lower volatility is better)
    # Convert parameters to cross-sectional Z-scores so they're comparable
    z_mom = (momentum.T - momentum.mean(axis=1)) / (momentum.std(axis=1) + 1e-8)
    z_mom = z_mom.T
    
    z_vol = -(volatility.T - volatility.mean(axis=1)) / (volatility.std(axis=1) + 1e-8)
    z_vol = z_vol.T
    
    # Combined factor score (equal weight)
    factor_score = (z_mom + z_vol) / 2
    factor_score = factor_score.fillna(0)
    
    # 4. Pick top 2 stocks each period
    # Rank 1 is best, 4 is worst
    ranks = factor_score.rank(axis=1, ascending=False)
    
    # Select stocks with rank <= 2
    portfolio_weights = (ranks <= 2).astype(float)
    # Normalize weights
    portfolio_weights = portfolio_weights.div(portfolio_weights.sum(axis=1).replace(0, 1), axis=0)
    
    # Shift weights by 1 day because we compute signals on day T, use for day T+1 returns
    portfolio_weights = portfolio_weights.shift(1).dropna()
    
    # Compute strategy returns
    strategy_returns = (portfolio_weights * returns.loc[portfolio_weights.index]).sum(axis=1)
    baseline_returns = returns.loc[portfolio_weights.index].mean(axis=1)
    
    cum_strategy = (1 + strategy_returns).cumprod()
    cum_baseline = (1 + baseline_returns).cumprod()
    
    results_df = pd.DataFrame({
        'Factor_Strategy': strategy_returns,
        'Baseline_EqualWeight': baseline_returns
    })
    results_df.to_csv('data/factor_portfolio.csv')
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cum_strategy.index, cum_strategy, label='Factor Strategy', color='blue')
    plt.plot(cum_baseline.index, cum_baseline, label='Baseline', color='gray', linestyle='--')
    plt.title('Factor Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/factor_portfolio.png')
    print("Saved plot to plots/factor_portfolio.png")

if __name__ == "__main__":
    main()
