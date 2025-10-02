import pandas as pd
import numpy as np
import warnings

# Part 1, 1
# scaling
TAU = 12

# read data
# change your own file path
file_path = 'C:/Users/Administrator/Desktop/uchi/fin market/36700 assignment/multi_asset_etf_data.xlsx'
#file_path =
df = pd.read_excel(file_path, sheet_name='total returns')
df = df.drop(columns=['Date'])
excess_returns = pd.read_excel(file_path, sheet_name='excess returns')
# drop QAI
excess_returns = excess_returns.drop(columns=['Date',"QAI"])

        
R_f = df['SHV']

# drop QAI
returns = df.drop(columns=['SHV','QAI'])

def calculate_annualized_stats(returns, tau=TAU, rf=R_f):
    # calculate annualized mean, volatility and sharp ratio
    
    # calculate period statistics (Monthly)
    mean_monthly = returns.mean()
    vol_monthly = returns.std()
    
    # calculate annualized statistics
    
    # 1.Mean Annualized
    mean_annualized = mean_monthly * tau

    # 2. Volatility Annualized
    vol_annualized = vol_monthly * np.sqrt(tau)
    
    # 3. Sharpe Ratio, no need to annualize
    sharpe = excess_returns.mean() / vol_monthly
    
    stats = pd.DataFrame({
        'Mean Return (Annualized)': mean_annualized,
        'Volatility (Annualized)': vol_annualized,
        'Sharpe Ratio': sharpe
    })

    return stats


# print results
results = calculate_annualized_stats(returns)

pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.width', 1000)

print(results)

#Part1, 2
def correlation_analysis(returns: pd.DataFrame):
  
    # 1. Compute correlation matrix
    corr_matrix = returns.corr()

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


# pirnt results
results = correlation_analysis(returns)

print("Correlation matrix:")
print(results["correlation_matrix"])

print(f"\nHighest correlation: {results['highest_pair']} = {results['highest_value']:.4f}")
print(f"Lowest correlation: {results['lowest_pair']} = {results['lowest_value']:.4f}")

#Part1, 3

def tangency_portfolio_analysis(returns: pd.DataFrame, excess_returns: pd.DataFrame, TAU: int = 12):
   
    # mean excess returns
    excess_mean_monthly = excess_returns.mean()
    excess_mean_annualized = excess_mean_monthly * TAU

    # covariance matrix
    sigma_matrix = returns.cov()
    sigma_inv = np.linalg.inv(sigma_matrix)

    # tangency portfolio weights
    unscaled_weight = np.dot(sigma_inv, excess_mean_annualized)
    scaling_constant = np.sum(unscaled_weight)
    weights = unscaled_weight / scaling_constant
    w_tan = pd.Series(weights.flatten(), index=returns.columns)

    # portfolio performance
    portfolio_return_monthly = np.dot(w_tan, returns.mean())
    portfolio_return_annualized = portfolio_return_monthly * TAU

    portfolio_volatility_monthly = np.sqrt(np.dot(w_tan.T, np.dot(sigma_matrix, w_tan)))
    portfolio_volatility_annualized = portfolio_volatility_monthly * np.sqrt(TAU)

    portfolio_sharpe_ratio = np.dot(w_tan, excess_mean_monthly) / portfolio_volatility_monthly

    # collect results
    results = {
        "weights": w_tan,
        "annualized_return": portfolio_return_annualized,
        "annualized_volatility": portfolio_volatility_annualized,
        "sharpe_ratio": portfolio_sharpe_ratio
    }
    
    return results

# print results
results = tangency_portfolio_analysis(returns, excess_returns, TAU=12)
print("Tangency portfolio weights:")
print(results["weights"])
print(f"Annualized return: {results['annualized_return']:.4f}")
print(f"Annualized volatility: {results['annualized_volatility']:.4f}")
print(f"Sharpe ratio: {results['sharpe_ratio']:.4f}")


#Part1, 4

# TIPS are dropped completely from the investment set

returns_drop_TIPS = returns.drop(columns = ['TIP'])
excess_returns_drop_TIPS = excess_returns.drop(columns = ['TIP'])

results = tangency_portfolio_analysis(returns_drop_TIPS, excess_returns_drop_TIPS, TAU = 12)

print("Tangency portfolio weights without TIPS:")
print(results["weights"])
print(f"Annualized return without TIPS: {results['annualized_return']:.4f}")
print(f"Annualized volatility without TIPS: {results['annualized_volatility']:.4f}")
print(f"Sharpe ratio without TIPS: {results['sharpe_ratio']:.4f}")

#The expected excess return to TIPS is adjusted to be 0.0012 higher than what the historic sample shows

adjusted_returns = returns
adjusted_returns['TIP'] += 0.0012
adjusted_excess_returns = excess_returns
adjusted_excess_returns['TIP'] += 0.0012

results = tangency_portfolio_analysis(adjusted_returns, adjusted_excess_returns, TAU = 12)

print("Tangency portfolio weights with TIPS adjusted:")
print(results["weights"])
print(f"Annualized return with TIPS adjusted: {results['annualized_return']:.4f}")
print(f"Annualized volatility with TIPS adjusted: {results['annualized_volatility']:.4f}")
print(f"Sharpe ratio with TIPS adjusted: {results['sharpe_ratio']:.4f}")