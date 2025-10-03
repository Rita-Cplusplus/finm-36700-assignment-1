import pandas as pd
import numpy as np
import warnings

# Import functions and data from main assignment
from assignment1 import (
    TAU, file_path, returns, excess_returns, target_mu,
    ew_weights_scaled, rp_weights_scaled, mv_weights_scaled,
    ew_performance, rp_performance, mv_performance,
    rescale_weights_to_target, calculate_portfolio_performance
)

# 4. Out-of-Sample Performance

# Load data with dates for time-based analysis
df_with_dates = pd.read_excel(file_path, sheet_name='total returns')
excess_returns_with_dates = pd.read_excel(file_path, sheet_name='excess returns')
excess_returns_with_dates = excess_returns_with_dates.drop(columns=['QAI'])

# Convert dates and create year column
df_with_dates['Date'] = pd.to_datetime(df_with_dates['Date'])
excess_returns_with_dates['Date'] = pd.to_datetime(excess_returns_with_dates['Date'])
df_with_dates['Year'] = df_with_dates['Date'].dt.year
excess_returns_with_dates['Year'] = excess_returns_with_dates['Date'].dt.year

def get_portfolio_weights(returns_data, excess_returns_data, target_mu, method='mv'):
    """
    Calculate portfolio weights for different methods
    """
    if method == 'ew':
        # Equally-weighted
        n_assets = len(returns_data.columns)
        weights = np.ones(n_assets) / n_assets
        weights = pd.Series(weights, index=returns_data.columns)
    elif method == 'rp':
        # Risk-parity (inverse variance)
        variances = returns_data.var()
        inv_variances = 1.0 / variances
        weights = inv_variances / inv_variances.sum()
    elif method == 'mv':
        # Mean-variance (tangency)
        excess_mean_monthly = excess_returns_data.mean()
        excess_mean_annualized = excess_mean_monthly * TAU
        sigma_matrix = returns_data.cov()
        sigma_inv = np.linalg.inv(sigma_matrix)
        unscaled_weight = np.dot(sigma_inv, excess_mean_annualized)
        scaling_constant = np.sum(unscaled_weight)
        weights = unscaled_weight / scaling_constant
        weights = pd.Series(weights.flatten(), index=returns_data.columns)
    elif method == 'reg':
        # Regularized (simple shrinkage of sample covariance)
        excess_mean_monthly = excess_returns_data.mean()
        excess_mean_annualized = excess_mean_monthly * TAU
        
        # Sample covariance matrix
        sample_cov = returns_data.cov()
        
        # Shrinkage towards identity matrix (simple regularization)
        shrinkage_factor = 0.1  # 10% shrinkage
        n_assets = len(returns_data.columns)
        identity_scaled = np.eye(n_assets) * np.trace(sample_cov) / n_assets
        regularized_cov = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * identity_scaled
        
        # Weights
        reg_inv = np.linalg.inv(regularized_cov)
        unscaled_weight = np.dot(reg_inv, excess_mean_annualized)
        scaling_constant = np.sum(unscaled_weight)
        weights = unscaled_weight / scaling_constant
        weights = pd.Series(weights.flatten(), index=returns_data.columns)
    
    # Rescale to target return
    portfolio_excess_return = np.dot(weights, excess_returns_data.mean())
    scaling_factor = target_mu / portfolio_excess_return
    weights = weights * scaling_factor
    
    return weights

def calculate_oos_performance(weights, returns_data, excess_returns_data, tau=TAU):
    """
    Calculate out-of-sample portfolio performance
    """
    # Monthly performance
    portfolio_return_monthly = np.dot(weights, returns_data.mean())
    portfolio_excess_return_monthly = np.dot(weights, excess_returns_data.mean())
    
    # Portfolio variance and volatility
    cov_matrix = returns_data.cov()
    portfolio_variance_monthly = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility_monthly = np.sqrt(portfolio_variance_monthly)
    
    # Annualized metrics
    portfolio_return_annualized = portfolio_return_monthly * tau
    portfolio_volatility_annualized = portfolio_volatility_monthly * np.sqrt(tau)
    portfolio_sharpe_ratio = portfolio_excess_return_monthly / portfolio_volatility_monthly if portfolio_volatility_monthly > 0 else 0
    
    return {
        'return': portfolio_return_annualized,
        'volatility': portfolio_volatility_annualized,
        'sharpe_ratio': portfolio_sharpe_ratio,
        'excess_return_monthly': portfolio_excess_return_monthly
    }

# 4.1 One-step Out-of-Sample Performance

print("4.1 One-step Out-of-Sample Performance")

# Split data: in-sample (through 2023) and out-of-sample (2024-2025)
in_sample_mask = df_with_dates['Year'] <= 2023
oos_mask = df_with_dates['Year'] >= 2024

# In-sample data (through 2023)
returns_in_sample = df_with_dates[in_sample_mask].drop(columns=['Date', 'Year', 'SHV', 'QAI'])
excess_returns_in_sample = excess_returns_with_dates[in_sample_mask].drop(columns=['Date', 'Year'])

# Out-of-sample data (2024-2025)
returns_oos = df_with_dates[oos_mask].drop(columns=['Date', 'Year', 'SHV', 'QAI'])
excess_returns_oos = excess_returns_with_dates[oos_mask].drop(columns=['Date', 'Year'])

methods = ['ew', 'rp', 'mv', 'reg']
method_names = ['Equally-Weighted', 'Risk-Parity', 'Mean-Variance', 'Regularized']

oos_results = {}

for method, name in zip(methods, method_names):
    # Get weights using in-sample data
    weights = get_portfolio_weights(returns_in_sample, excess_returns_in_sample, target_mu, method)
    
    # In-sample performance
    is_performance = calculate_oos_performance(weights, returns_in_sample, excess_returns_in_sample)
    
    # Out-of-sample performance
    oos_performance = calculate_oos_performance(weights, returns_oos, excess_returns_oos)
    
    oos_results[name] = {
        'weights': weights,
        'in_sample': is_performance,
        'out_of_sample': oos_performance
    }

# Display results
print(f"\nIn-Sample Performance (through 2023):")
is_comparison = pd.DataFrame({
    name: [results['in_sample']['return'], results['in_sample']['volatility'], results['in_sample']['sharpe_ratio']]
    for name, results in oos_results.items()
}, index=['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio'])

# Set pandas options for full display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(is_comparison.round(4))

print(f"\nOut-of-Sample Performance (2024-2025):")
oos_comparison = pd.DataFrame({
    name: [results['out_of_sample']['return'], results['out_of_sample']['volatility'], results['out_of_sample']['sharpe_ratio']]
    for name, results in oos_results.items()
}, index=['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio'])
print(oos_comparison.round(4))

# 4.2 Rolling Out-of-Sample Performance

print("4.2 Rolling Out-of-Sample Performance")

def rolling_oos_analysis(df_with_dates, excess_returns_with_dates, start_year=2016, end_year=2025):
    """
    Perform rolling out-of-sample analysis
    """
    rolling_results = {name: {'returns': [], 'volatilities': [], 'sharpe_ratios': []} 
                      for name in method_names}
    
    for year in range(start_year, end_year):
        # Training data
        train_mask = df_with_dates['Year'] <= (year - 1)
        # Test data: current year
        test_mask = df_with_dates['Year'] == year
            
        # Prepare training data
        returns_train = df_with_dates[train_mask].drop(columns=['Date', 'Year', 'SHV', 'QAI'])
        excess_returns_train = excess_returns_with_dates[train_mask].drop(columns=['Date', 'Year'])
        
        # Prepare test data
        returns_test = df_with_dates[test_mask].drop(columns=['Date', 'Year', 'SHV', 'QAI'])
        excess_returns_test = excess_returns_with_dates[test_mask].drop(columns=['Date', 'Year'])
        
        for method, name in zip(methods, method_names):
            # Get weights
            weights = get_portfolio_weights(returns_train, excess_returns_train, target_mu, method)
            
            # performance
            test_performance = calculate_oos_performance(weights, returns_test, excess_returns_test)
            
            rolling_results[name]['returns'].append(test_performance['return'])
            rolling_results[name]['volatilities'].append(test_performance['volatility'])
            rolling_results[name]['sharpe_ratios'].append(test_performance['sharpe_ratio'])
    
    return rolling_results

# Perform rolling analysis
rolling_results = rolling_oos_analysis(df_with_dates, excess_returns_with_dates)

# Average performance across all rolling periods
print(f"\nRolling OOS Performance (2016-2024, averaged):")
rolling_summary = {}
for name in method_names:
    avg_return = np.mean(rolling_results[name]['returns'])
    avg_volatility = np.mean(rolling_results[name]['volatilities'])
    avg_sharpe = np.mean(rolling_results[name]['sharpe_ratios'])
    rolling_summary[name] = [avg_return, avg_volatility, avg_sharpe]

rolling_comparison = pd.DataFrame(rolling_summary, 
                                index=['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio'])
print(rolling_comparison.round(4))

# Find best performing method
best_oos_sharpe = rolling_comparison.loc['Sharpe Ratio'].max()
best_oos_method = rolling_comparison.loc['Sharpe Ratio'].idxmax()

print(f"\nRolling OOS Analysis:")
print(f"Best average Sharpe ratio: {best_oos_method} ({best_oos_sharpe:.4f})")
print(f"Number of rolling periods analyzed: {len(rolling_results[method_names[0]]['returns'])}")

# Comparison
print(f"\nOOS vs In-Sample Comparison:")
for name in method_names:
    is_sharpe = oos_results[name]['in_sample']['sharpe_ratio']
    oos_sharpe = oos_results[name]['out_of_sample']['sharpe_ratio']
    rolling_sharpe = rolling_comparison.loc['Sharpe Ratio', name]
    print(f"{name}:")
    print(f"  In-Sample Sharpe: {is_sharpe:.4f}")
    print(f"  One-step OOS Sharpe: {oos_sharpe:.4f}")
    print(f"  Rolling OOS Sharpe: {rolling_sharpe:.4f}")

print(f"\nOut-of-Sample Analysis Summary:")
print("- In-sample performance may not predict out-of-sample performance")
print("- Rolling analysis provides more robust performance estimates")
print("- Regularization can help prevent overfitting in small samples")

print("5. EXTRA: Without a Riskless Asset")

def minimum_variance_portfolio(excess_returns):
    """
    Calculate the minimum variance portfolio without a risk-free asset.
    w_mvp = (Σ^(-1) * 1) / (1' * Σ^(-1) * 1)
    """
    # covariance matrix
    cov_matrix = excess_returns.cov() * TAU
    ones = np.ones((len(cov_matrix), 1))
    inv_cov = np.linalg.inv(cov_matrix.values)
    
    # w_mvp = (Σ^(-1) * 1) / (1' * Σ^(-1) * 1)
    numerator = inv_cov @ ones
    denominator = ones.T @ inv_cov @ ones
    weights_mvp = (numerator / denominator).flatten()
    
    weights_mvp = pd.Series(weights_mvp, index=excess_returns.columns)
    
    # Portfolio statistics
    portfolio_return = np.sum(weights_mvp * excess_returns.mean()) * TAU
    portfolio_variance = weights_mvp.T @ cov_matrix @ weights_mvp
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    return weights_mvp, portfolio_return, portfolio_volatility

def mean_variance_no_riskfree(excess_returns, target_return):
    """
    Calculate mean-variance portfolio without risk-free asset.
    min w'Σw subject to w'μ = μ_target and w'1 = 1
    """
    # Statistics
    mu = excess_returns.mean() * TAU  # Annualized mean returns
    cov_matrix = excess_returns.cov() * TAU  # Annualized covariance matrix
    
    # Create vectors
    ones = np.ones(len(mu))
    mu_vec = mu.values
    
    inv_cov = np.linalg.inv(cov_matrix.values)
    
    # Required matrix elements
    A = mu_vec.T @ inv_cov @ mu_vec  # μ'Σ⁻¹μ
    B = mu_vec.T @ inv_cov @ ones    # μ'Σ⁻¹1 
    C = ones.T @ inv_cov @ ones      # 1'Σ⁻¹1
    
    matrix = np.array([[A, B], [B, C]])
    rhs = np.array([target_return, 1.0])
    
    # Solve for λ₁ and λ₂
    lambdas = np.linalg.solve(matrix, rhs)
    lambda1, lambda2 = lambdas[0], lambdas[1]
    
    # Portfolio weights
    weights = lambda1 * (inv_cov @ mu_vec) + lambda2 * (inv_cov @ ones)
    
    weights = pd.Series(weights, index=excess_returns.columns)
    
    # Portfolio statistics
    portfolio_return = np.sum(weights * mu)
    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return weights, portfolio_return, portfolio_volatility, sharpe_ratio

# 5.1 Minimum Variance Portfolio
print("\n5.1 Minimum Variance Portfolio")

# Minimum variance portfolio
mvp_weights, mvp_return, mvp_volatility = minimum_variance_portfolio(excess_returns)

print("\nMinimum Variance Portfolio weights:")
print(mvp_weights)
print(f"\nMinimum Variance Portfolio Statistics:")
print(f"Annualized return: {mvp_return:.4f}")
print(f"Annualized volatility: {mvp_volatility:.4f}")
print(f"Sharpe ratio: {mvp_return/mvp_volatility:.4f}")

# 5.2 Mean-Variance Frontier without Risk-free Asset
print("\n5.2 Mean-Variance Frontier without Risk-free Asset")

# Test different target returns
target_returns = [0.06, 0.08, 0.10, 0.12, 0.14]
frontier_results = []

print(f"\nMean-Variance Efficient Portfolios (No Risk-free Asset):")
print(f"{'Target Return':<15}{'Volatility':<12}{'Sharpe Ratio':<12}")

for target_ret in target_returns:
    weights, ret, vol, sharpe = mean_variance_no_riskfree(excess_returns, target_ret)
    frontier_results.append({
        'target_return': target_ret,
        'actual_return': ret,
        'volatility': vol,
        'sharpe_ratio': sharpe,
        'weights': weights
    })
    print(f"{target_ret:<15.2%}{vol:<12.4f}{sharpe:<12.4f}")

# 5.3 Comparison with Risk-free Asset Case
print("\n5.3 Comparison with Risk-free Asset Case")

# Tangency portfolio for comparison (from earlier)
def tangency_portfolio_comparison(excess_returns):
    """Recalculate tangency portfolio for comparison"""
    mu = excess_returns.mean() * TAU
    cov_matrix = excess_returns.cov() * TAU
    inv_cov = np.linalg.inv(cov_matrix.values)
    
    # Tangency portfolio weights
    weights = inv_cov @ mu.values
    weights = weights / np.sum(weights)
    weights = pd.Series(weights, index=excess_returns.columns)
    
    # Portfolio statistics
    portfolio_return = np.sum(weights * mu)
    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return weights, portfolio_return, portfolio_volatility, sharpe_ratio

# Get tangency portfolio for comparison
tangency_weights, tangency_return, tangency_vol, tangency_sharpe = tangency_portfolio_comparison(excess_returns)

# set target return for comparison(eg. 10%)
comparison_target_return = 0.10

print(f"\nComparison at target return = {comparison_target_return:.1%}:")

# Get no-risk-free portfolio at 10% target return
norisk_weights, norisk_return, norisk_vol, norisk_sharpe = mean_variance_no_riskfree(
    excess_returns, comparison_target_return)

# For with risk-free asset, we need to scale the tangency portfolio to achieve 10% return
# Scale factor = target_return / tangency_return
scale_factor = comparison_target_return / tangency_return
with_risk_free_weights = tangency_weights * scale_factor
with_risk_free_return = comparison_target_return
with_risk_free_vol = tangency_vol * scale_factor
with_risk_free_sharpe = tangency_sharpe  # Sharpe ratio unchanged when scaling

print(f"\nWith Risk-free Asset (Scaled Tangency Portfolio):")
print(f"Return: {with_risk_free_return:.4f}")
print(f"Volatility: {with_risk_free_vol:.4f}")
print(f"Sharpe Ratio: {with_risk_free_sharpe:.4f}")
print(f"Scale factor: {scale_factor:.4f}")

print(f"\nWithout Risk-free Asset:")
print(f"Return: {norisk_return:.4f}")
print(f"Volatility: {norisk_vol:.4f}")
print(f"Sharpe Ratio: {norisk_sharpe:.4f}")

print(f"\nWeight differences (No Risk-free - With Risk-free):")
weight_diff = norisk_weights - with_risk_free_weights
print(weight_diff.round(4))

print(f"\nComparison Summary:")
print(f"- Volatility difference: {(norisk_vol - with_risk_free_vol):.4f}")
print(f"- Sharpe ratio difference: {(norisk_sharpe - with_risk_free_sharpe):.4f}")
print(f"- Without risk-free asset has {'higher' if norisk_vol > with_risk_free_vol else 'lower'} volatility")
print(f"- Without risk-free asset has {'higher' if norisk_sharpe > with_risk_free_sharpe else 'lower'} Sharpe ratio")

# 5.4 Analysis Summary
print("\n5.4 Analysis Summary")
print(f"- Minimum variance portfolio return: {mvp_return:.4f}")
print(f"- Minimum variance portfolio volatility: {mvp_volatility:.4f}")
print(f"- Without risk-free asset, portfolio composition changes nonlinearly with target return")
print(f"- The efficient frontier is a hyperbola rather than a straight line from risk-free rate")
print(f"- At the same target return, portfolios have similar but not identical compositions")

# Show the nonlinear relationship by comparing weights at different target returns
print(f"\nWeight Variation with Target Return (showing nonlinear relationship):")
print(f"Asset weights for different target returns:")

# Create a DataFrame to show how weights change
weight_comparison = pd.DataFrame()
for i, result in enumerate(frontier_results):
    weight_comparison[f"{result['target_return']:.1%}"] = result['weights']

print(weight_comparison.round(4))

print(f"\nThis demonstrates the nonlinear relationship between target return and portfolio weights")
print(f"when no risk-free asset is available, unlike the simple scaling with a risk-free rate.")

# 6. EXTRA: Bayesian Allocation

print("6. EXTRA: Bayesian Allocation")

def calculate_regularized_allocation(excess_returns, target_mu=0.01):
    """
    Calculate regularized (REG) allocation using a regularized covariance matrix.
    
    The regularized covariance matrix is:
    Σ_reg = 0.5 * Σ + 0.5 * D
    
    Weights are proportional to: Σ_reg^(-1) * μ
    """
    # Sample covariance matrix (monthly)
    cov_matrix = excess_returns.cov()
    
    # Extract diagonal matrix of variances
    diagonal_matrix = pd.DataFrame(np.diag(np.diag(cov_matrix)), 
                                 index=cov_matrix.index, 
                                 columns=cov_matrix.columns)

    # Σ_reg = 0.5 * Σ + 0.5 * D
    regularized_cov = 0.5 * cov_matrix + 0.5 * diagonal_matrix
    
    # Mean excess returns (monthly)
    mu = excess_returns.mean()
    
    # Weights proportional to Σ_reg^(-1) * μ
    inv_reg_cov = np.linalg.inv(regularized_cov.values)
    weights = inv_reg_cov @ mu.values
    
    # Convert to Series
    weights = pd.Series(weights, index=excess_returns.columns)
    
    return weights, regularized_cov

# 6.1 Regularized Allocation Implementation
print("\n6.1 Regularized Allocation Implementation")

# Regularized allocation
reg_weights, reg_cov_matrix = calculate_regularized_allocation(excess_returns)

print("\nRegularized (REG) weights (before scaling):")
print(reg_weights.round(4))
print(f"\nWeights sum: {reg_weights.sum():.4f}")

# Scale to achieve target monthly excess return
reg_weights_scaled = rescale_weights_to_target(reg_weights, returns, excess_returns, target_mu)

print(f"\nScaled REG weights (target μ = {target_mu}):")
print(reg_weights_scaled.round(4))

# Performance
reg_performance = calculate_portfolio_performance(reg_weights_scaled, returns, excess_returns)

print(f"\nPerformance:")
print(f"Annualized Return: {reg_performance['return']:.4f}")
print(f"Annualized Volatility: {reg_performance['volatility']:.4f}")
print(f"Sharpe Ratio: {reg_performance['sharpe_ratio']:.4f}")
print(f"Monthly Excess Return: {reg_performance['excess_return_monthly']:.4f}")

# 6.2 Comparison with Other Methods
print("\n6.2 Comparison with Other Methods")

# Recalculate all methods for comparison
methods = {
    'Equally-Weighted': ew_weights_scaled,
    'Risk-Parity': rp_weights_scaled, 
    'Mean-Variance': mv_weights_scaled,
    'Regularized': reg_weights_scaled
}

# Performance for all methods
performance_comparison = {}
for method_name, weights in methods.items():
    perf = calculate_portfolio_performance(weights, returns, excess_returns)
    performance_comparison[method_name] = perf

# Create comparison table
comparison_df = pd.DataFrame({
    method: {
        'Annualized Return': perf['return'],
        'Annualized Volatility': perf['volatility'], 
        'Sharpe Ratio': perf['sharpe_ratio']
    }
    for method, perf in performance_comparison.items()
})

print("\nPerformance Comparison (Including Regularized):")
print(comparison_df.round(4))

# 6.3 Regularization Analysis
print("\n6.3 Regularization Analysis")

# Compare original vs regularized covariance matrices
original_cov = excess_returns.cov()

print(f"\nCovariance Matrix Analysis:")
print(f"Original covariance matrix condition number: {np.linalg.cond(original_cov.values):.2f}")
print(f"Regularized covariance matrix condition number: {np.linalg.cond(reg_cov_matrix.values):.2f}")

# Show how correlations are affected by regularization
original_corr = original_cov / np.sqrt(np.outer(np.diag(original_cov), np.diag(original_cov)))
reg_corr = reg_cov_matrix / np.sqrt(np.outer(np.diag(reg_cov_matrix), np.diag(reg_cov_matrix)))

print(f"\nCorrelation Changes Due to Regularization:")
print(f"Original correlations - range: [{original_corr.values[np.triu_indices_from(original_corr, k=1)].min():.3f}, {original_corr.values[np.triu_indices_from(original_corr, k=1)].max():.3f}]")
print(f"Regularized correlations - range: [{reg_corr.values[np.triu_indices_from(reg_corr, k=1)].min():.3f}, {reg_corr.values[np.triu_indices_from(reg_corr, k=1)].max():.3f}]")

# Compare weight concentrations
print(f"\nWeight Concentration Analysis:")
for method_name, weights in methods.items():
    # Herfindahl index (concentration measure)
    herfindahl = np.sum(weights**2)
    # Effective number of assets
    effective_assets = 1 / herfindahl if herfindahl > 0 else 0
    print(f"{method_name:<15}: Herfindahl = {herfindahl:.4f}, Effective assets = {effective_assets:.2f}")

# 6.4 Bayesian Interpretation
print("\n6.4 Bayesian Interpretation")
print(f"- The regularized covariance matrix shrinks correlations toward zero")
print(f"- This represents a Bayesian prior that assets are less correlated than observed")
print(f"- Regularization reduces estimation error in the covariance matrix")
print(f"- The 50% shrinkage factor balances between sample information and prior beliefs")

# Show specific examples of how correlations change
print(f"\nExample Correlation Changes:")
asset_pairs = [('SPY', 'EFA'), ('SPY', 'IEF'), ('HYG', 'IEF'), ('DBC', 'IEF')]

for asset1, asset2 in asset_pairs:
    original_corr_val = original_cov.loc[asset1, asset2] / np.sqrt(original_cov.loc[asset1, asset1] * original_cov.loc[asset2, asset2])
    reg_corr_val = reg_cov_matrix.loc[asset1, asset2] / np.sqrt(reg_cov_matrix.loc[asset1, asset1] * reg_cov_matrix.loc[asset2, asset2])
    change = reg_corr_val - original_corr_val
    print(f"{asset1}-{asset2}: {original_corr_val:.3f} → {reg_corr_val:.3f} (change: {change:+.3f})")

# 6.5 Updated Summary
print("\n6.5 Updated Allocation Summary")

print(f"\nFinal Comparison Including Regularized Method:")

# Find best performers
best_sharpe = max(comparison_df.loc['Sharpe Ratio'])
best_sharpe_method = comparison_df.loc['Sharpe Ratio'].idxmax()

lowest_vol = min(comparison_df.loc['Annualized Volatility'])
lowest_vol_method = comparison_df.loc['Annualized Volatility'].idxmin()

highest_return = max(comparison_df.loc['Annualized Return'])
highest_return_method = comparison_df.loc['Annualized Return'].idxmax()

print(f"Best Sharpe Ratio: {best_sharpe_method} ({best_sharpe:.4f})")
print(f"Lowest Volatility: {lowest_vol_method} ({lowest_vol:.4f})")
print(f"Highest Return: {highest_return_method} ({highest_return:.4f})")

print(f"\nTarget monthly excess return verification:")
for method_name, perf in performance_comparison.items():
    print(f"{method_name}: {perf['excess_return_monthly']:.6f}")
print(f"Target: {target_mu:.6f}")

print(f"\nKey Insights:")
print(f"- Regularization typically reduces portfolio concentration")
print(f"- Bayesian shrinkage makes the covariance matrix more stable")
print(f"- REG method balances between sample data and prior beliefs")
print(f"- All methods achieve the target monthly excess return of {target_mu:.4f}")

reg_sharpe = performance_comparison['Regularized']['sharpe_ratio']
mv_sharpe = performance_comparison['Mean-Variance']['sharpe_ratio']
print(f"- Regularized Sharpe ratio ({reg_sharpe:.4f}) vs Mean-Variance ({mv_sharpe:.4f}): {'better' if reg_sharpe > mv_sharpe else 'worse'}")

print("7. EXTRA: Inefficient Tangency")

# Load data including QAI
excess_returns_with_qai = pd.read_excel(file_path, sheet_name='excess returns')
excess_returns_with_qai = excess_returns_with_qai.drop(columns=['Date'])
returns_with_qai = pd.read_excel(file_path, sheet_name='total returns')
returns_with_qai = returns_with_qai.drop(columns=['Date', 'SHV'])

print(f"\n7.1 Data with QAI included")
print(f"Assets included: {list(excess_returns_with_qai.columns)}")
print(f"Number of assets: {len(excess_returns_with_qai.columns)}")

# 7.1 Tangency Portfolio Analysis with QAI
print(f"\n7.2 Tangency Portfolio Analysis with QAI")

def tangency_portfolio_with_qai(excess_returns):
    """Calculate tangency portfolio including QAI"""
    mu = excess_returns.mean() * TAU
    cov_matrix = excess_returns.cov() * TAU
    
    # Tangency portfolio weights
    inv_cov = np.linalg.inv(cov_matrix.values)
    weights = inv_cov @ mu.values
    weights = weights / np.sum(weights)
    weights = pd.Series(weights, index=excess_returns.columns)
    
    # Calculate portfolio statistics
    portfolio_return = np.sum(weights * mu)
    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return weights, portfolio_return, portfolio_volatility, sharpe_ratio

# Calculate tangency portfolio with QAI
qai_tangency_weights, qai_tangency_return, qai_tangency_vol, qai_tangency_sharpe = tangency_portfolio_with_qai(excess_returns_with_qai)

print(f"\nTangency portfolio with QAI:")
print(qai_tangency_weights.round(4))
print(f"\nTangency portfolio statistics:")
print(f"Annualized return: {qai_tangency_return:.4f}")
print(f"Annualized volatility: {qai_tangency_vol:.4f}")
print(f"Sharpe ratio: {qai_tangency_sharpe:.4f}")

print(f"Tangency portfolio has negative return: {qai_tangency_return < 0}")

# 7.2 Optimal Allocation Strategy
print(f"\n7.3 Optimal Allocation Strategy")
print("Tangency portfolio is efficient, no need to short.")

# Use the tangency portfolio weights as optimal weights
optimal_weights = qai_tangency_weights

# 7.3 Re-do Section 3 Analysis with QAI
print(f"\n7.4 Section 3 Analysis with QAI")

# Helper functions for analysis with QAI
def rescale_weights_to_target_qai(weights, returns_data, excess_returns_data, target_mu):
    """Rescale weights to achieve target return (QAI version)"""
    portfolio_excess_return = np.dot(weights, excess_returns_data.mean())
    scaling_factor = target_mu / portfolio_excess_return
    scaled_weights = weights * scaling_factor
    return scaled_weights

def calculate_portfolio_performance_qai(weights, returns_data, excess_returns_data, tau=TAU):
    """Calculate portfolio performance (QAI version)"""
    portfolio_return_monthly = np.dot(weights, returns_data.mean())
    portfolio_excess_return_monthly = np.dot(weights, excess_returns_data.mean())
    
    cov_matrix = returns_data.cov()
    portfolio_variance_monthly = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility_monthly = np.sqrt(portfolio_variance_monthly)
    
    portfolio_return_annualized = portfolio_return_monthly * tau
    portfolio_volatility_annualized = portfolio_volatility_monthly * np.sqrt(tau)
    portfolio_sharpe_ratio = portfolio_excess_return_monthly / portfolio_volatility_monthly
    
    return {
        'return': portfolio_return_annualized,
        'volatility': portfolio_volatility_annualized,
        'sharpe_ratio': portfolio_sharpe_ratio,
        'excess_return_monthly': portfolio_excess_return_monthly
    }

# Equally-weighted portfolio with QAI
n_assets_qai = len(excess_returns_with_qai.columns)
ew_weights_qai = pd.Series(np.ones(n_assets_qai) / n_assets_qai, index=excess_returns_with_qai.columns)
ew_weights_qai_scaled = rescale_weights_to_target_qai(ew_weights_qai, returns_with_qai, excess_returns_with_qai, target_mu)
ew_performance_qai = calculate_portfolio_performance_qai(ew_weights_qai_scaled, returns_with_qai, excess_returns_with_qai)

print(f"\nEqually-Weighted portfolio with QAI:")
print(f"Scaled EW weights (target μ = {target_mu}):")
print(ew_weights_qai_scaled.round(4))
print(f"Performance:")
print(f"Annualized Return: {ew_performance_qai['return']:.4f}")
print(f"Annualized Volatility: {ew_performance_qai['volatility']:.4f}")
print(f"Sharpe Ratio: {ew_performance_qai['sharpe_ratio']:.4f}")

# Risk-parity portfolio with QAI
variances_qai = returns_with_qai.var()
rp_weights_qai = 1 / variances_qai
rp_weights_qai = rp_weights_qai / rp_weights_qai.sum()
rp_weights_qai_scaled = rescale_weights_to_target_qai(rp_weights_qai, returns_with_qai, excess_returns_with_qai, target_mu)
rp_performance_qai = calculate_portfolio_performance_qai(rp_weights_qai_scaled, returns_with_qai, excess_returns_with_qai)

print(f"\nRisk-Parity portfolio with QAI:")
print(f"Scaled RP weights (target μ = {target_mu}):")
print(rp_weights_qai_scaled.round(4))
print(f"Performance:")
print(f"Annualized Return: {rp_performance_qai['return']:.4f}")
print(f"Annualized Volatility: {rp_performance_qai['volatility']:.4f}")
print(f"Sharpe Ratio: {rp_performance_qai['sharpe_ratio']:.4f}")

# Mean-Variance portfolio with QAI (using optimal allocation)
mv_weights_qai_scaled = rescale_weights_to_target_qai(optimal_weights, returns_with_qai, excess_returns_with_qai, target_mu)
mv_performance_qai = calculate_portfolio_performance_qai(mv_weights_qai_scaled, returns_with_qai, excess_returns_with_qai)

print(f"\nMean-Variance portfolio with QAI (optimal allocation):")
print(f"Scaled MV weights (target μ = {target_mu}):")
print(mv_weights_qai_scaled.round(4))
print(f"Performance:")
print(f"Annualized Return: {mv_performance_qai['return']:.4f}")
print(f"Annualized Volatility: {mv_performance_qai['volatility']:.4f}")
print(f"Sharpe Ratio: {mv_performance_qai['sharpe_ratio']:.4f}")

# Regularized portfolio with QAI
def calculate_regularized_allocation_qai(excess_returns_data):
    """Calculate regularized allocation with QAI"""
    cov_matrix = excess_returns_data.cov()
    diagonal_matrix = pd.DataFrame(np.diag(np.diag(cov_matrix)), 
                                 index=cov_matrix.index, 
                                 columns=cov_matrix.columns)
    regularized_cov = 0.5 * cov_matrix + 0.5 * diagonal_matrix
    mu = excess_returns_data.mean()
    inv_reg_cov = np.linalg.inv(regularized_cov.values)
    weights = inv_reg_cov @ mu.values
    weights = pd.Series(weights, index=excess_returns_data.columns)
    return weights

reg_weights_qai = calculate_regularized_allocation_qai(excess_returns_with_qai)
reg_weights_qai_scaled = rescale_weights_to_target_qai(reg_weights_qai, returns_with_qai, excess_returns_with_qai, target_mu)
reg_performance_qai = calculate_portfolio_performance_qai(reg_weights_qai_scaled, returns_with_qai, excess_returns_with_qai)

print(f"\nRegularized portfolio with QAI:")
print(f"Scaled REG weights (target μ = {target_mu}):")
print(reg_weights_qai_scaled.round(4))
print(f"Performance:")
print(f"Annualized Return: {reg_performance_qai['return']:.4f}")
print(f"Annualized Volatility: {reg_performance_qai['volatility']:.4f}")
print(f"Sharpe Ratio: {reg_performance_qai['sharpe_ratio']:.4f}")

# 7.4 Comparison: With vs Without QAI
print(f"\n7.5 Comparison: With vs Without QAI")

methods_qai = {
    'Equally-Weighted': ew_performance_qai,
    'Risk-Parity': rp_performance_qai,
    'Mean-Variance': mv_performance_qai,
    'Regularized': reg_performance_qai
}

methods_original = {
    'Equally-Weighted': ew_performance,
    'Risk-Parity': rp_performance,
    'Mean-Variance': mv_performance,
    'Regularized': reg_performance
}

print(f"\nPerformance Comparison:")
print(f"{'Method':<15} {'Without QAI':<12} {'With QAI':<12} {'Difference':<12}")
print(f"{'(Sharpe Ratio)':<15} {'Sharpe':<12} {'Sharpe':<12} {'(pp)':<12}")
print("-" * 55)

for method in methods_qai.keys():
    sharpe_without = methods_original[method]['sharpe_ratio']
    sharpe_with = methods_qai[method]['sharpe_ratio']
    difference = sharpe_with - sharpe_without
    
    print(f"{method:<15} {sharpe_without:<12.4f} {sharpe_with:<12.4f} {difference:+8.4f}")

# Volatility comparison
print(f"\nVolatility Comparison:")
print(f"{'Method':<15} {'Without QAI':<12} {'With QAI':<12} {'Difference':<12}")
print(f"{'(Volatility)':<15} {'Vol':<12} {'Vol':<12} {'(pp)':<12}")
print("-" * 55)

for method in methods_qai.keys():
    vol_without = methods_original[method]['volatility']
    vol_with = methods_qai[method]['volatility']
    difference = vol_with - vol_without
    
    print(f"{method:<15} {vol_without:<12.4f} {vol_with:<12.4f} {difference:+8.4f}")

# 7.5 Impact Analysis
print(f"\n7.6 Impact Analysis")

print(f"\nQAI Impact Summary:")
qai_return = excess_returns_with_qai['QAI'].mean() * TAU
qai_volatility = excess_returns_with_qai['QAI'].std() * np.sqrt(TAU)
qai_sharpe = qai_return / qai_volatility

print(f"QAI standalone statistics:")
print(f"  Annualized return: {qai_return:.4f}")
print(f"  Annualized volatility: {qai_volatility:.4f}")
print(f"  Sharpe ratio: {qai_sharpe:.4f}")

# Check correlations with QAI
qai_correlations = excess_returns_with_qai.corr()['QAI'].drop('QAI')
print(f"\nQAI correlations with other assets:")
print(qai_correlations.sort_values(ascending=False).round(4))

# Analyze QAI weight in different portfolios
print(f"\nQAI allocation in different portfolios:")
print(f"Equally-Weighted: {ew_weights_qai_scaled['QAI']:.4f} ({ew_weights_qai_scaled['QAI']/ew_weights_qai_scaled.sum()*100:.1f}%)")
print(f"Risk-Parity: {rp_weights_qai_scaled['QAI']:.4f} ({rp_weights_qai_scaled['QAI']/rp_weights_qai_scaled.sum()*100:.1f}%)")
print(f"Mean-Variance: {mv_weights_qai_scaled['QAI']:.4f} ({mv_weights_qai_scaled['QAI']/mv_weights_qai_scaled.sum()*100:.1f}%)")
print(f"Regularized: {reg_weights_qai_scaled['QAI']:.4f} ({reg_weights_qai_scaled['QAI']/reg_weights_qai_scaled.sum()*100:.1f}%)")

print(f"\nKey Findings:")
# Find the method with biggest improvement/deterioration
sharpe_changes = {method: methods_qai[method]['sharpe_ratio'] - methods_original[method]['sharpe_ratio'] 
                 for method in methods_qai.keys()}
best_improvement = max(sharpe_changes, key=sharpe_changes.get)
worst_change = min(sharpe_changes, key=sharpe_changes.get)

print(f"- Best improvement from QAI inclusion: {best_improvement} (+{sharpe_changes[best_improvement]:.4f} Sharpe)")
print(f"- Worst impact from QAI inclusion: {worst_change} ({sharpe_changes[worst_change]:+.4f} Sharpe)")
print(f"- QAI tangency portfolio is on efficient frontier")
print(f"- Optimal strategy is to long the tangency portfolio")

# Check if QAI improves diversification
avg_vol_change = np.mean([methods_qai[method]['volatility'] - methods_original[method]['volatility'] 
                         for method in methods_qai.keys()])
print(f"- Average volatility change: {avg_vol_change:+.4f} ({'reduces' if avg_vol_change < 0 else 'increases'} risk)")

print(f"\nConclusion: QAI addition {'significantly changes' if max([abs(x) for x in sharpe_changes.values()]) > 0.1 else 'does not significantly change'} the portfolio analysis.")

