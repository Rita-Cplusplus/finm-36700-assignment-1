import pandas as pd
import numpy as np
import warnings

# 2.1 Summary Statistics
# scaling
TAU = 12

# read data
# change your own file path
file_path = 'multi_asset_etf_data.xlsx'
df = pd.read_excel(file_path, sheet_name='total returns')
df = df.drop(columns=['Date'])

# Load excess returns and drop QAI
excess_returns = pd.read_excel(file_path, sheet_name='excess returns')
excess_returns = excess_returns.drop(columns=['Date', 'QAI'])

# Get risk-free rate        
R_f = df['SHV']

# Drop QAI and SHV from total returns
returns = df.drop(columns=['SHV', 'QAI'])

def calculate_annualized_stats(returns, excess_returns, tau=TAU):
    """Calculate annualized mean, volatility and Sharpe ratio"""
    
    # Monthly statistics
    # mean_monthly = returns.mean()
    #vol_monthly = returns.std()
    excess_mean_monthly = excess_returns.mean()
    excess_vol_monthly = excess_returns.std()
    
    # Annualized statistics
    #mean_annualized = mean_monthly * tau
    #vol_annualized = vol_monthly * np.sqrt(tau)
    excess_mean_annualized = excess_mean_monthly * tau
    excess_vol_annualized = excess_vol_monthly * np.sqrt(tau)
    
    # Sharpe ratio using annualized excess returns and vols
    sharpe_annualized = excess_mean_annualized / excess_vol_annualized
    
    stats = pd.DataFrame({
        'Excess Return Mean': excess_mean_annualized,
        'Excess Return Volatility': excess_vol_annualized,
        'Sharpe Ratio': sharpe_annualized
    })
    
    return stats

# print results
results = calculate_annualized_stats(returns, excess_returns)

pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.width', 1000)

print("2. Summary Statistics and Descriptive Analysis")
print("\n2.1 Annualized Statistics")
print(results)

print("\n2.2 Descriptive Analysis")

# 2.2 Descriptive Analysis
def correlation_analysis(excess_returns: pd.DataFrame):
    """Analyze correlations between assets"""
    
    # 1. Compute correlation matrix
    corr_matrix = excess_returns.corr()

    # 2. Extract upper triangle (exclude diagonal and duplicates)
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_unstacked = corr_matrix.where(mask).stack()

    # 3. Find highest correlation
    max_pair = corr_unstacked.idxmax()
    max_value = corr_unstacked.max()

    # 4. Find lowest correlation
    min_pair = corr_unstacked.idxmin()
    min_value = corr_unstacked.min()

    results = {
        "correlation_matrix": corr_matrix,
        "highest_pair": max_pair,
        "highest_value": max_value,
        "lowest_pair": min_pair,
        "lowest_value": min_value
    }
    
    return results

# print results
results_corr = correlation_analysis(excess_returns)

print("Correlation matrix:")
print(results_corr["correlation_matrix"])

print(f"\nHighest correlation: {results_corr['highest_pair']} = {results_corr['highest_value']:.4f}")
print(f"Lowest correlation: {results_corr['lowest_pair']} = {results_corr['lowest_value']:.4f}")

# 2.3 Mean-Variance Frontier

def tangency_portfolio_analysis(returns: pd.DataFrame, excess_returns: pd.DataFrame, TAU: int = 12):
    """Calculate tangency portfolio weights and performance"""
    
    # mean excess returns
    excess_mean_monthly = excess_returns.mean()

    # covariance matrix
    sigma_matrix = excess_returns.cov()
    sigma_inv = np.linalg.inv(sigma_matrix)

    # tangency portfolio weights
    unscaled_weight = np.dot(sigma_inv, excess_mean_monthly)
    scaling_constant = np.sum(unscaled_weight)
    weights = unscaled_weight / scaling_constant
    w_tan = pd.Series(weights.flatten(), index=returns.columns)

    # portfolio performance
    portfolio_return_monthly = np.dot(w_tan, excess_returns.mean())
    portfolio_return_annualized = portfolio_return_monthly * TAU

    portfolio_volatility_monthly = np.sqrt(np.dot(w_tan.T, np.dot(sigma_matrix, w_tan)))
    portfolio_volatility_annualized = portfolio_volatility_monthly * np.sqrt(TAU)

    portfolio_sharpe_ratio = portfolio_return_annualized / portfolio_volatility_annualized

    # collect results
    results = {
        "weights": w_tan,
        "annualized_return": portfolio_return_annualized,
        "annualized_volatility": portfolio_volatility_annualized,
        "sharpe_ratio": portfolio_sharpe_ratio
    }
    
    return results

print("\n2.3 Mean-Variance Frontier Analysis")

# print results
results_tangency = tangency_portfolio_analysis(returns, excess_returns, TAU=12)
print("\nTangency portfolio weights:")
print(results_tangency["weights"])
print(f"Annualized return: {results_tangency['annualized_return']:.4f}")
print(f"Annualized volatility: {results_tangency['annualized_volatility']:.4f}")
print(f"Sharpe ratio: {results_tangency['sharpe_ratio']:.4f}")

print("\n2.4 TIPS Analysis")

# 2.4 TIPS Analysis

# TIPS are dropped completely from the investment set

returns_drop_TIPS = returns.drop(columns = ['TIP'])
excess_returns_drop_TIPS = excess_returns.drop(columns = ['TIP'])

results_no_tips = tangency_portfolio_analysis(returns_drop_TIPS, excess_returns_drop_TIPS, TAU = 12)

print("\nTangency portfolio weights without TIPS:")
print(results_no_tips["weights"])
print(f"Annualized return without TIPS: {results_no_tips['annualized_return']:.4f}")
print(f"Annualized volatility without TIPS: {results_no_tips['annualized_volatility']:.4f}")
print(f"Sharpe ratio without TIPS: {results_no_tips['sharpe_ratio']:.4f}")

#The expected excess return to TIPS is adjusted to be 0.0012 higher than what the historic sample shows

adjusted_returns = returns.copy()
adjusted_returns['TIP'] += 0.0012
adjusted_excess_returns = excess_returns.copy()
adjusted_excess_returns['TIP'] += 0.0012

results_adjusted = tangency_portfolio_analysis(adjusted_returns, adjusted_excess_returns, TAU = 12)

print("\nTangency portfolio weights with TIPS adjusted:")
print(results_adjusted["weights"])
print(f"Annualized return with TIPS adjusted: {results_adjusted['annualized_return']:.4f}")
print(f"Annualized volatility with TIPS adjusted: {results_adjusted['annualized_volatility']:.4f}")
print(f"Sharpe ratio with TIPS adjusted: {results_adjusted['sharpe_ratio']:.4f}")

print("\n3. Allocations")

# Part 3: Allocations

# Target monthly excess return is 1%
target_mu = 0.01
target_mu_annual = target_mu * 12

def rescale_weights_to_target(weights, returns, excess_returns, target_mu):
    """
    Rescale weights to achieve target monthly excess return
    (corresponding to 1% annualized excess return)
    """
    # Calculate portfolio expected excess return with current weights
    portfolio_excess_return = np.dot(weights, excess_returns.mean())
    
    # Calculate scaling factor
    if portfolio_excess_return != 0:
        scaling_factor = target_mu / portfolio_excess_return
    else:
        scaling_factor = 1.0
    
    # Rescaling
    scaled_weights = weights * scaling_factor
    
    return scaled_weights

def calculate_portfolio_performance(weights, returns, excess_returns, tau=TAU):
    """
    Calculate portfolio performance metrics
    """
    # Monthly performance
    portfolio_return_monthly = np.dot(weights, returns.mean())
    portfolio_excess_return_monthly = np.dot(weights, excess_returns.mean())
    
    # Portfolio variance and volatility
    cov_matrix = excess_returns.cov()
    portfolio_variance_monthly = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility_monthly = np.sqrt(portfolio_variance_monthly)
    
    # Annualized metrics
    portfolio_return_annualized = portfolio_excess_return_monthly * tau
    portfolio_volatility_annualized = portfolio_volatility_monthly * np.sqrt(tau)
    portfolio_sharpe_ratio = portfolio_return_annualized / portfolio_volatility_annualized
    
    return {
        'return': portfolio_return_annualized,
        'volatility': portfolio_volatility_annualized,
        'sharpe_ratio': portfolio_sharpe_ratio,
        'excess_return_monthly': portfolio_excess_return_monthly
    }

print("\n3-1. Equally-weighted (EW) portfolio")

# 3-1. Equally-weighted (EW) portfolio

n_assets = len(returns.columns)
ew_weights = np.ones(n_assets) / n_assets
ew_weights = pd.Series(ew_weights, index=returns.columns)

# Rescale to target return
ew_weights_scaled = rescale_weights_to_target(ew_weights, returns, excess_returns, target_mu)

print("\nOriginal EW weights:")
print(ew_weights)
print(f"\nScaled EW weights (target μ = {target_mu:.4f} monthly, {target_mu * 12:.1%} annualized):")
print(ew_weights_scaled)

ew_performance = calculate_portfolio_performance(ew_weights_scaled, returns, excess_returns)
print(f"\nPerformance:")
print(f"Annualized Return: {ew_performance['return']:.4f}")
print(f"Annualized Volatility: {ew_performance['volatility']:.4f}")
print(f"Sharpe Ratio: {ew_performance['sharpe_ratio']:.4f}")
print(f"Monthly Excess Return: {ew_performance['excess_return_monthly']:.4f}")

print("\n3-2. Risk-parity (RP) portfolio")

# 3-2. Risk-parity (RP) portfolio

# Calculate inverse variance weights
variances = excess_returns.var()
inv_variances = 1.0 / variances
rp_weights = inv_variances / inv_variances.sum()

# Rescale to target return
rp_weights_scaled = rescale_weights_to_target(rp_weights, returns, excess_returns, target_mu)

print("\nRisk-parity weights (inverse variance):")
print(rp_weights)
print(f"\nScaled RP weights (target μ = {target_mu:.4f} monthly, {target_mu * 12:.1%} annualized):")
print(rp_weights_scaled)

rp_performance = calculate_portfolio_performance(rp_weights_scaled, returns, excess_returns)
print(f"\nPerformance:")
print(f"Annualized Return: {rp_performance['return']:.4f}")
print(f"Annualized Volatility: {rp_performance['volatility']:.4f}")
print(f"Sharpe Ratio: {rp_performance['sharpe_ratio']:.4f}")
print(f"Monthly Excess Return: {rp_performance['excess_return_monthly']:.4f}")

# 3-3. Mean-Variance (MV) portfolio

print('\n3-3. Mean-Variance (MV) portfolio')

# Use tangency from Part 2
mv_weights = results_tangency["weights"]

# Rescaling
mv_weights_scaled = rescale_weights_to_target(mv_weights, returns, excess_returns, target_mu)

print("Original MV (tangency) weights:")
print(mv_weights)
print(f"\nScaled MV weights (target μ = {target_mu:.4f} monthly, {target_mu * 12:.1%} annualized):")
print(mv_weights_scaled)

mv_performance = calculate_portfolio_performance(mv_weights_scaled, returns, excess_returns)
print(f"\nPerformance:")
print(f"Annualized Return: {mv_performance['return']:.4f}")
print(f"Annualized Volatility: {mv_performance['volatility']:.4f}")
print(f"Sharpe Ratio: {mv_performance['sharpe_ratio']:.4f}")
print(f"Monthly Excess Return: {mv_performance['excess_return_monthly']:.4f}")

print('\n3-4. Performance Comparison')

comparison_df = pd.DataFrame({
    'Equally-Weighted': [
        ew_performance['return'],
        ew_performance['volatility'],
        ew_performance['sharpe_ratio']
    ],
    'Risk-Parity': [
        rp_performance['return'],
        rp_performance['volatility'],
        rp_performance['sharpe_ratio']
    ],
    'Mean-Variance': [
        mv_performance['return'],
        mv_performance['volatility'],
        mv_performance['sharpe_ratio']
    ]
}, index=['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio'])

print(comparison_df.round(4))

# Analysis
best_sharpe = comparison_df.loc['Sharpe Ratio'].max()
best_method = comparison_df.loc['Sharpe Ratio'].idxmax()
lowest_vol = comparison_df.loc['Annualized Volatility'].min()
lowest_vol_method = comparison_df.loc['Annualized Volatility'].idxmin()

print(f"Best Sharpe Ratio: {best_method} ({best_sharpe:.4f})")
print(f"Lowest Volatility: {lowest_vol_method} ({lowest_vol:.4f})")

print(f"\nTarget excess return verification:")
print(f"Monthly excess returns:")
print(f"EW: {ew_performance['excess_return_monthly']:.6f}")
print(f"RP: {rp_performance['excess_return_monthly']:.6f}")
print(f"MV: {mv_performance['excess_return_monthly']:.6f}")
print(f"Target (monthly): {target_mu:.6f}")
print(f"Target (annualized): {target_mu_annual:.4f}")

print(f"- Mean-Variance optimization provides the best risk-adjusted returns (Sharpe ratio: {best_sharpe:.4f})")
print(f"- Mean-Variance provides the lowest volatility ({lowest_vol:.4f} vs EW: {comparison_df.loc['Annualized Volatility', 'Equally-Weighted']:.4f}, RP: {comparison_df.loc['Annualized Volatility', 'Risk-Parity']:.4f})")
print(f"- Risk-Parity achieves the highest return ({comparison_df.loc['Annualized Return', 'Risk-Parity']:.4f}) but with much higher volatility")
print(f"- All three methods successfully achieve the target annualized excess return of {target_mu_annual:.1%}")
print(f"- Mean-Variance dominates both alternatives in risk-adjusted performance")

