# FINM 36700 Assignment 1 - Problems 4-7 Output

## 4. Out-of-Sample Performance

### 4.1 One-step Out-of-Sample Performance

#### In-Sample Performance (through 2023):
```
                       Equally-Weighted  Risk-Parity  Mean-Variance  Regularized
Annualized Return                0.1281       0.1328         0.1228       0.1225
Annualized Volatility            0.2710       0.2526         0.0788       0.0842
Sharpe Ratio                     0.1278       0.1372         0.4393       0.4113
```

#### Out-of-Sample Performance (2024-2025):
```
                       Equally-Weighted  Risk-Parity  Mean-Variance  Regularized
Annualized Return                0.1962       0.2008         0.1468       0.1428
Annualized Volatility            0.1735       0.2196         0.1088       0.1012
Sharpe Ratio                     0.1490       0.0436         0.2906       0.3138
```

### 4.2 Rolling Out-of-Sample Performance

#### Rolling OOS Performance (2016-2024, averaged):
```
                       Equally-Weighted  Risk-Parity  Mean-Variance  Regularized
Annualized Return                0.1965       0.1236         0.0846       0.0787
Annualized Volatility            0.2441       0.1950         0.0779       0.0808
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

#### Scaled REG weights (target μ = 0.01):
```
BWX   -0.6210
DBC   -0.1315
EEM   -0.0780
EFA    0.0694
HYG    0.3774
IEF    0.3938
IYR    0.0379
PSP    0.0245
SPY    0.5952
TIP    0.4418
```

**Performance:**
- Annualized Return: 0.1277
- Annualized Volatility: 0.1056
- Sharpe Ratio: 0.3281
- Monthly Excess Return: 0.0100

### 6.2 Comparison with Other Methods

#### Performance Comparison (Including Regularized):
```
                       Equally-Weighted  Risk-Parity  Mean-Variance  Regularized
Annualized Return                0.1383       0.1497         0.1264       0.1277
Annualized Volatility            0.2695       0.2646         0.0816       0.1056
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
Equally-Weighted: Herfindahl = 0.7048, Effective assets = 1.42
Risk-Parity    : Herfindahl = 3.6304, Effective assets = 0.28
Mean-Variance  : Herfindahl = 2.5380, Effective assets = 0.39
Regularized    : Herfindahl = 1.2629, Effective assets = 0.79
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
- **Lowest Volatility:** Mean-Variance (0.0816)
- **Highest Return:** Risk-Parity (0.1497)

#### Target monthly excess return verification:
- Equally-Weighted: 0.010000
- Risk-Parity: 0.010000
- Mean-Variance: 0.010000
- Regularized: 0.010000
- Target: 0.010000

**Key Insights:**
- Regularization typically reduces portfolio concentration
- Bayesian shrinkage makes the covariance matrix more stable
- REG method balances between sample data and prior beliefs
- All methods achieve the target monthly excess return of 0.0100
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
**Scaled EW weights (target μ = 0.01):**
```
BWX   0.2546
DBC   0.2546
EEM   0.2546
EFA   0.2546
HYG   0.2546
IEF   0.2546
IYR   0.2546
PSP   0.2546
QAI   0.2546
SPY   0.2546
TIP   0.2546
```
**Performance:**
- Annualized Return: 0.1393
- Annualized Volatility: 0.2699
- Sharpe Ratio: 0.1283

#### Risk-Parity portfolio with QAI:
**Scaled RP weights (target μ = 0.01):**
```
BWX   0.4282
DBC   0.1085
EEM   0.0961
EFA   0.1309
HYG   0.5170
IEF   0.7272
IYR   0.1053
PSP   0.0657
QAI   1.2233
SPY   0.1467
TIP   1.1342
```
**Performance:**
- Annualized Return: 0.1523
- Annualized Volatility: 0.2646
- Sharpe Ratio: 0.1309

#### Mean-Variance portfolio with QAI (optimal allocation):
**Scaled MV weights (target μ = 0.01):**
```
BWX   -0.6355
DBC   -0.0117
EEM    0.0887
EFA    0.0400
HYG    0.2737
IEF    0.9360
IYR   -0.2475
PSP   -0.1784
QAI   -1.4145
SPY    1.1146
TIP    0.1382
```
**Performance:**
- Annualized Return: 0.1207
- Annualized Volatility: 0.0773
- Sharpe Ratio: 0.4481

#### Regularized portfolio with QAI:
**Scaled REG weights (target μ = 0.01):**
```
BWX   -0.6215
DBC   -0.1317
EEM   -0.0784
EFA    0.0690
HYG    0.3767
IEF    0.3936
IYR    0.0377
PSP    0.0242
QAI    0.0104
SPY    0.5946
TIP    0.4412
```
**Performance:**
- Annualized Return: 0.1277
- Annualized Volatility: 0.1056
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
Equally-Weighted 0.2695       0.2699        +0.0005
Risk-Parity     0.2646       0.2646        -0.0000
Mean-Variance   0.0816       0.0773        -0.0043
Regularized     0.1056       0.1056        +0.0001
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
- Average volatility change: -0.0009 (reduces risk)

**Conclusion:** QAI addition does not significantly change the portfolio analysis.

