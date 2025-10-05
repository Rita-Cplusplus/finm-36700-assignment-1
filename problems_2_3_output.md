# FINM 36700 Assignment 1 - Problems 2-3 Output

## 2. Summary Statistics and Descriptive Analysis

### 2.1 Annualized Statistics
```
     Excess Return Mean  Excess Return Volatility  Sharpe Ratio
BWX             -0.0077                    0.0828       -0.0932
DBC             -0.0053                    0.1666       -0.0318
EEM              0.0293                    0.1762        0.1665
EFA              0.0618                    0.1509        0.4094
HYG              0.0414                    0.0759        0.5449
IEF              0.0164                    0.0634        0.2586
IYR              0.0749                    0.1687        0.4441
PSP              0.0926                    0.2134        0.4338
SPY              0.1281                    0.1428        0.8971
TIP              0.0205                    0.0511        0.4011
```

### 2.2 Descriptive Analysis

#### Correlation Matrix
```
       BWX     DBC    EEM    EFA    HYG     IEF    IYR    PSP    SPY    TIP
BWX 1.0000  0.1911 0.6217 0.6028 0.6026  0.5809 0.5526 0.5267 0.4400 0.6752
DBC 0.1911  1.0000 0.5117 0.5009 0.4619 -0.3002 0.2805 0.4533 0.4322 0.1090
EEM 0.6217  0.5117 1.0000 0.8199 0.6912  0.0267 0.5841 0.7501 0.6878 0.3788
EFA 0.6028  0.5009 0.8199 1.0000 0.7872  0.0426 0.6993 0.8953 0.8459 0.3948
HYG 0.6026  0.4619 0.6912 0.7872 1.0000  0.1873 0.7394 0.8122 0.7935 0.5386
IEF 0.5809 -0.3002 0.0267 0.0426 0.1873  1.0000 0.3165 0.0224 0.0008 0.7541
IYR 0.5526  0.2805 0.5841 0.6993 0.7394  0.3165 1.0000 0.7498 0.7547 0.5987
PSP 0.5267  0.4533 0.7501 0.8953 0.8122  0.0224 0.7498 1.0000 0.8917 0.4080
SPY 0.4400  0.4322 0.6878 0.8459 0.7935  0.0008 0.7547 0.8917 1.0000 0.3816
TIP 0.6752  0.1090 0.3788 0.3948 0.5386  0.7541 0.5987 0.4080 0.3816 1.0000
```

**Key Findings:**
- Highest correlation: ('EFA', 'PSP') = 0.8953
- Lowest correlation: ('DBC', 'IEF') = -0.3002

### 2.3 Mean-Variance Frontier Analysis

#### Tangency Portfolio (Full)
```
BWX   -0.8506
DBC   -0.0716
EEM    0.0264
EFA    0.0687
HYG    0.2906
IEF    0.8812
IYR   -0.2466
PSP   -0.3330
SPY    1.0596
TIP    0.1753
```
- Annualized return: 0.1285
- Annualized volatility: 0.0875
- Sharpe ratio: 1.4692

#### 2.4 TIPS Analysis

##### Tangency Portfolio without TIPS
```
BWX   -0.8793
DBC   -0.0637
EEM    0.0325
EFA    0.0602
HYG    0.3163
IEF    1.0200
IYR   -0.2474
PSP   -0.3394
SPY    1.1010
```
- Annualized return without TIPS: 0.1327
- Annualized volatility without TIPS: 0.0905
- Sharpe ratio without TIPS: 1.4675

##### Tangency Portfolio with TIPS Adjusted
```
BWX   -0.6697
DBC   -0.1214
EEM   -0.0118
EFA    0.1226
HYG    0.1284
IEF    0.0047
IYR   -0.2411
PSP   -0.2926
SPY    0.7985
TIP    1.2825
```
- Annualized return with TIPS adjusted: 0.1204
- Annualized volatility with TIPS adjusted: 0.0746
- Sharpe ratio with TIPS adjusted: 1.6124

## 3. Allocations

### 3-1. Equally-weighted (EW) portfolio

#### Original EW weights:
```
BWX   0.1000
DBC   0.1000
EEM   0.1000
EFA   0.1000
HYG   0.1000
IEF   0.1000
IYR   0.1000
PSP   0.1000
SPY   0.1000
TIP   0.1000
```

#### Scaled EW weights (target μ = 0.0100 monthly, 12.0% annualized):
```
BWX   0.2655
DBC   0.2655
EEM   0.2655
EFA   0.2655
HYG   0.2655
IEF   0.2655
IYR   0.2655
PSP   0.2655
SPY   0.2655
TIP   0.2655
```

**Performance:**
- Annualized Return: 0.1200
- Annualized Volatility: 0.2692
- Sharpe Ratio: 0.4457
- Monthly Excess Return: 0.0100

### 3-2. Risk-parity (RP) portfolio

#### Risk-parity weights (inverse variance):
```
BWX   0.1248
DBC   0.0308
EEM   0.0276
EFA   0.0376
HYG   0.1484
IEF   0.2126
IYR   0.0301
PSP   0.0188
SPY   0.0419
TIP   0.3274
```

#### Scaled RP weights (target μ = 0.0100 monthly, 12.0% annualized):
```
BWX   0.5406
DBC   0.1336
EEM   0.1194
EFA   0.1627
HYG   0.6427
IEF   0.9206
IYR   0.1302
PSP   0.0814
SPY   0.1816
TIP   1.4181
```

**Performance:**
- Annualized Return: 0.1200
- Annualized Volatility: 0.2639
- Sharpe Ratio: 0.4547
- Monthly Excess Return: 0.0100

### 3-3. Mean-Variance (MV) portfolio

#### Original MV (tangency) weights:
```
BWX   -0.8506
DBC   -0.0716
EEM    0.0264
EFA    0.0687
HYG    0.2906
IEF    0.8812
IYR   -0.2466
PSP   -0.3330
SPY    1.0596
TIP    0.1753
```

#### Scaled MV weights (target μ = 0.0100 monthly, 12.0% annualized):
```
BWX   -0.7942
DBC   -0.0669
EEM    0.0247
EFA    0.0641
HYG    0.2713
IEF    0.8228
IYR   -0.2302
PSP   -0.3109
SPY    0.9894
TIP    0.1637
```

**Performance:**
- Annualized Return: 0.1200
- Annualized Volatility: 0.0817
- Sharpe Ratio: 1.4692
- Monthly Excess Return: 0.0100

### 3-4. Performance Comparison

```
                       Equally-Weighted  Risk-Parity  Mean-Variance
Annualized Return                0.1200       0.1200         0.1200
Annualized Volatility            0.2692       0.2639         0.0817
Sharpe Ratio                     0.4457       0.4547         1.4692
```

**Best Sharpe Ratio:** Mean-Variance (1.4692)  
**Lowest Volatility:** Mean-Variance (0.0817)

### Target Excess Return Verification:
**Monthly excess returns:**
- EW: 0.010000
- RP: 0.010000
- MV: 0.010000
- Target (monthly): 0.010000
- Target (annualized): 0.1200

## Key Insights

- **Mean-Variance optimization** provides the best risk-adjusted returns (Sharpe ratio: 1.4692)
- **Mean-Variance** provides the lowest volatility (0.0817 vs EW: 0.2692, RP: 0.2639)
- **Risk-Parity** achieves the highest return (0.1200) but with much higher volatility
- All three methods successfully achieve the target annualized excess return of 12.0%
- **Mean-Variance dominates** both alternatives in risk-adjusted performance
