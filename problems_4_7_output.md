# FINM 36700 Assignment 1 - Problems 4-7 Output

## 4. Out-of-Sample Performance

### 4.1 One-step Out-of-Sample Performance

#### In-Sample Performance (through 2023):
```
                       Equally-Weighted  Risk-Parity  Mean-Variance  Regularized
Annualized Return                0.0107       0.0111         0.0102       0.0102
Annualized Volatility            0.0226       0.0210         0.0066       0.0070
Sharpe Ratio                     0.1278       0.1372         0.4393       0.4113
```

#### Out-of-Sample Performance (2024-2025):
```
                       Equally-Weighted  Risk-Parity  Mean-Variance  Regularized
Annualized Return                0.0163       0.0167         0.0122       0.0119
Annualized Volatility            0.0145       0.0183         0.0091       0.0084
Sharpe Ratio                     0.1490       0.0436         0.2906       0.3138
```

### 4.2 Rolling Out-of-Sample Performance

#### Rolling OOS Performance (2016-2024, averaged):
```
                       Equally-Weighted  Risk-Parity  Mean-Variance  Regularized
Annualized Return                0.0164       0.0103         0.0070       0.0066
Annualized Volatility            0.0203       0.0162         0.0065       0.0067
Sharpe Ratio                     0.3371       0.2424         0.3695       0.3444
```

**Rolling OOS Analysis:**
- Best average Sharpe ratio: Mean-Variance (0.3695)
- Number of rolling periods analyzed: 9

#### OOS vs In-Sample Comparison:
**Equally-Weighted:**
- In-Sample Sharpe: 0.1278
- One-step OOS Sharpe: 0.1490
- Rolling OOS Sharpe: 0.3371

**Risk-Parity:**
- In-Sample Sharpe: 0.1372
- One-step OOS Sharpe: 0.0436
- Rolling OOS Sharpe: 0.2424

**Mean-Variance:**
- In-Sample Sharpe: 0.4393
- One-step OOS Sharpe: 0.2906
- Rolling OOS Sharpe: 0.3695

**Regularized:**
- In-Sample Sharpe: 0.4113
- One-step OOS Sharpe: 0.3138
- Rolling OOS Sharpe: 0.3444

### Out-of-Sample Analysis Summary:
- In-sample performance may not predict out-of-sample performance
- Rolling analysis provides more robust performance estimates
- Regularization can help prevent overfitting in small samples

---

## 5. EXTRA: Without a Riskless Asset

### 5.1 Minimum Variance Portfolio

#### Minimum Variance Portfolio weights:
```
BWX   -0.1052
DBC    0.0631
EEM   -0.0226
EFA    0.0558
HYG    0.3555
IEF    0.2741
IYR   -0.1730
PSP   -0.1259
SPY    0.1416
TIP    0.5364
```

**Minimum Variance Portfolio Statistics:**
- Annualized return: 0.0270
- Annualized volatility: 0.0401
- Sharpe ratio: 0.6734

### 5.2 Mean-Variance Frontier without Risk-free Asset

#### Mean-Variance Efficient Portfolios (No Risk-free Asset):
```
Target Return  Volatility  Sharpe Ratio
---------------------------------------
6.00%          0.0474      1.2660      
8.00%          0.0571      1.4022      
10.00%         0.0688      1.4536      
12.00%         0.0817      1.4682      
14.00%         0.0954      1.4679      
```

### 5.3 Comparison with Risk-free Asset Case

#### Comparison at target return = 10.0%:

**With Risk-free Asset (Scaled Tangency Portfolio):**
- Return: 0.1000
- Volatility: 0.0681
- Sharpe Ratio: 1.4692
- Scale factor: 0.7781

**Without Risk-free Asset:**
- Return: 0.1000
- Volatility: 0.0688
- Sharpe Ratio: 1.4536

#### Weight differences (No Risk-free - With Risk-free):
```
BWX    0.0207
DBC    0.0220
EEM   -0.0079
EFA    0.0116
HYG    0.0827
IEF    0.0250
IYR   -0.0340
PSP   -0.0157
SPY   -0.0227
TIP    0.1403
```

**Comparison Summary:**
- Volatility difference: 0.0007
- Sharpe ratio difference: -0.0156
- Without risk-free asset has higher volatility
- Without risk-free asset has lower Sharpe ratio

### 5.4 Analysis Summary

**Key Findings:**
- Minimum variance portfolio return: 0.0270
- Minimum variance portfolio volatility: 0.0401
- Without risk-free asset, portfolio composition changes nonlinearly with target return
- The efficient frontier is a hyperbola rather than a straight line from risk-free rate
- At the same target return, portfolios have similar but not identical compositions

#### Weight Variation with Target Return (showing nonlinear relationship):
```
Asset weights for different target returns:
       6.0%    8.0%   10.0%   12.0%   14.0%
BWX -0.3475 -0.4943 -0.6412 -0.7881 -0.9349
DBC  0.0193 -0.0072 -0.0338 -0.0603 -0.0869
EEM -0.0066  0.0030  0.0127  0.0223  0.0320
EFA  0.0600  0.0625  0.0651  0.0676  0.0701
HYG  0.3344  0.3216  0.3088  0.2961  0.2833
IEF  0.4715  0.5911  0.7106  0.8302  0.9498
IYR -0.1969 -0.2114 -0.2259 -0.2404 -0.2549
PSP -0.1932 -0.2340 -0.2748 -0.3156 -0.3564
SPY  0.4400  0.6209  0.8017  0.9826  1.1634
TIP  0.4190  0.3479  0.2767  0.2056  0.1345
```

This demonstrates the nonlinear relationship between target return and portfolio weights when no risk-free asset is available, unlike the simple scaling with a risk-free rate.

---

## 6. EXTRA: Bayesian Allocation

### 6.1 Regularized Allocation Implementation

#### Regularized (REG) weights (before scaling):
```
BWX   -6.2517
DBC   -1.3239
EEM   -0.7856
EFA    0.6990
HYG    3.7996
IEF    3.9649
IYR    0.3815
PSP    0.2470
SPY    5.9918
TIP    4.4478
```
Weights sum: 11.1704

#### Scaled REG weights (target μ = 0.0008 monthly, 1.0% annualized):
```
BWX   -0.0518
DBC   -0.0110
EEM   -0.0065
EFA    0.0058
HYG    0.0315
IEF    0.0328
IYR    0.0032
PSP    0.0020
SPY    0.0496
TIP    0.0368
```

**Performance:**
- Annualized Return: 0.0106
- Annualized Volatility: 0.0088
- Sharpe Ratio: 0.3281
- Monthly Excess Return: 0.0008

### 6.2 Comparison with Other Methods

#### Performance Comparison (Including Regularized):
```
                       Equally-Weighted  Risk-Parity  Mean-Variance  Regularized
Annualized Return                0.0115       0.0125         0.0105       0.0106
Annualized Volatility            0.0225       0.0221         0.0068       0.0088
Sharpe Ratio                     0.1286       0.1309         0.4244       0.3281
```

### 6.3 Regularization Analysis

#### Covariance Matrix Analysis:
- Original covariance matrix condition number: 357.67
- Regularized covariance matrix condition number: 48.89

#### Correlation Changes Due to Regularization:
- Original correlations - range: [-0.300, 0.895]
- Regularized correlations - range: [-0.150, 0.448]

#### Weight Concentration Analysis:
```
Equally-Weighted: Herfindahl = 0.0049, Effective assets = 204.30
Risk-Parity    : Herfindahl = 0.0252, Effective assets = 39.67
Mean-Variance  : Herfindahl = 0.0176, Effective assets = 56.74
Regularized    : Herfindahl = 0.0088, Effective assets = 114.02
```

### 6.4 Bayesian Interpretation

**Bayesian Shrinkage Analysis:**
- The regularized covariance matrix shrinks correlations toward zero
- This represents a Bayesian prior that assets are less correlated than observed
- Regularization reduces estimation error in the covariance matrix
- The 50% shrinkage factor balances between sample information and prior beliefs

#### Example Correlation Changes:
- SPY-EFA: 0.846 → 0.423 (change: -0.423)
- SPY-IEF: 0.001 → 0.000 (change: -0.000)
- HYG-IEF: 0.187 → 0.094 (change: -0.094)
- DBC-IEF: -0.300 → -0.150 (change: +0.150)

### 6.5 Updated Allocation Summary

#### Final Comparison Including Regularized Method:
- **Best Sharpe Ratio:** Mean-Variance (0.4244)
- **Lowest Volatility:** Mean-Variance (0.0068)
- **Highest Return:** Risk-Parity (0.0125)

#### Target excess return verification:
- Equally-Weighted: 0.000833
- Risk-Parity: 0.000833
- Mean-Variance: 0.000833
- Regularized: 0.000833
- Target (monthly): 0.000833
- Target (annualized): 0.0100

**Key Insights:**
- Regularization typically reduces portfolio concentration
- Bayesian shrinkage makes the covariance matrix more stable
- REG method balances between sample data and prior beliefs
- All methods achieve the target annualized excess return of 1.0%
- Regularized Sharpe ratio (0.3281) vs Mean-Variance (0.4244): worse

---

## 7. EXTRA: Inefficient Tangency

### 7.1 Data with QAI included
- Assets included: ['BWX', 'DBC', 'EEM', 'EFA', 'HYG', 'IEF', 'IYR', 'PSP', 'QAI', 'SPY', 'TIP']
- Number of assets: 11

### 7.2 Tangency Portfolio Analysis with QAI

#### Tangency portfolio with QAI:
```
BWX    -6.1174
DBC    -0.1123
EEM     0.8536
EFA     0.3853
HYG     2.6349
IEF     9.0106
IYR    -2.3821
PSP    -1.7168
QAI   -13.6160
SPY    10.7298
TIP     1.3306
```

**Tangency portfolio statistics:**
- Annualized return: 1.1551
- Annualized volatility: 0.7444
- Sharpe ratio: 1.5517
- Tangency portfolio has negative return: False

### 7.3 Optimal Allocation Strategy
Tangency portfolio is efficient, no need to short.

### 7.4 Section 3 Analysis with QAI

#### Equally-Weighted portfolio with QAI:
**Scaled EW weights (target μ = 0.0008 monthly, 1.0% annualized):**
```
BWX   0.0212
DBC   0.0212
EEM   0.0212
EFA   0.0212
HYG   0.0212
IEF   0.0212
IYR   0.0212
PSP   0.0212
QAI   0.0212
SPY   0.0212
TIP   0.0212
```
**Performance:**
- Annualized Return: 0.0116
- Annualized Volatility: 0.0225
- Sharpe Ratio: 0.1283

#### Risk-Parity portfolio with QAI:
**Scaled RP weights (target μ = 0.0008 monthly, 1.0% annualized):**
```
BWX   0.0357
DBC   0.0090
EEM   0.0080
EFA   0.0109
HYG   0.0431
IEF   0.0606
IYR   0.0088
PSP   0.0055
QAI   0.1019
SPY   0.0122
TIP   0.0945
```
**Performance:**
- Annualized Return: 0.0127
- Annualized Volatility: 0.0221
- Sharpe Ratio: 0.1309

#### Mean-Variance portfolio with QAI (optimal allocation):
**Scaled MV weights (target μ = 0.0008 monthly, 1.0% annualized):**
```
BWX   -0.0530
DBC   -0.0010
EEM    0.0074
EFA    0.0033
HYG    0.0228
IEF    0.0780
IYR   -0.0206
PSP   -0.0149
QAI   -0.1179
SPY    0.0929
TIP    0.0115
```
**Performance:**
- Annualized Return: 0.0101
- Annualized Volatility: 0.0064
- Sharpe Ratio: 0.4481

#### Regularized portfolio with QAI:
**Scaled REG weights (target μ = 0.0008 monthly, 1.0% annualized):**
```
BWX   -0.0518
DBC   -0.0110
EEM   -0.0065
EFA    0.0058
HYG    0.0314
IEF    0.0328
IYR    0.0031
PSP    0.0020
QAI    0.0009
SPY    0.0496
TIP    0.0368
```
**Performance:**
- Annualized Return: 0.0106
- Annualized Volatility: 0.0088
- Sharpe Ratio: 0.3279

### 7.5 Comparison: With vs Without QAI

#### Performance Comparison:
```
Method          Without QAI  With QAI     Difference  
(Sharpe Ratio)  Sharpe       Sharpe       (pp)        
-------------------------------------------------------
Equally-Weighted 0.1286       0.1283        -0.0002
Risk-Parity     0.1309       0.1309        +0.0000
Mean-Variance   0.4244       0.4481        +0.0237
Regularized     0.3281       0.3279        -0.0002
```

#### Volatility Comparison:
```
Method          Without QAI  With QAI     Difference  
(Volatility)    Vol          Vol          (pp)        
-------------------------------------------------------
Equally-Weighted 0.0225       0.0225        +0.0000
Risk-Parity     0.0221       0.0221        -0.0000
Mean-Variance   0.0068       0.0064        -0.0004
Regularized     0.0088       0.0088        +0.0000
```

### 7.6 Impact Analysis

#### QAI Impact Summary:
**QAI standalone statistics:**
- Annualized return: 0.0193
- Annualized volatility: 0.0491
- Sharpe ratio: 0.3938

#### QAI correlations with other assets:
```
PSP   0.8734
SPY   0.8668
EFA   0.8479
HYG   0.8079
EEM   0.7747
IYR   0.7185
BWX   0.6303
TIP   0.5167
DBC   0.4753
IEF   0.1798
```

#### QAI allocation in different portfolios:
- Equally-Weighted: 0.2546 (9.1%)
- Risk-Parity: 1.2233 (26.1%)
- Mean-Variance: -1.4145 (-1361.6%)
- Regularized: 0.0104 (0.9%)

**Key Findings:**
- Best improvement from QAI inclusion: Mean-Variance (+0.0237 Sharpe)
- Worst impact from QAI inclusion: Equally-Weighted (-0.0002 Sharpe)
- QAI tangency portfolio is on efficient frontier
- Optimal strategy is to long the tangency portfolio
- Average volatility change: -0.0009 (reduces risk by 0.09 basis points on average)

**Conclusion:** QAI addition provides modest improvements to Mean-Variance optimization by reducing extreme allocations, while having negligible impact on other diversified strategies. The corrected 1% annualized excess return target produces realistic portfolio allocations and performance metrics.

