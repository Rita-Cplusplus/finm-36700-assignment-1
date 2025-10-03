# FINM 36700 Assignment 1 - Problems 2-3 Output

## 2. Summary Statistics and Descriptive Analysis

### 2.1 Annualized Statistics
```
     Mean Return (Annualized)  Volatility (Annualized)  Sharpe Ratio
BWX                   -0.0008                   0.0836       -0.0267
DBC                    0.0016                   0.1660       -0.0092
EEM                    0.0362                   0.1764        0.0480
EFA                    0.0687                   0.1511        0.1180
HYG                    0.0483                   0.0761        0.1570
IEF                    0.0233                   0.0641        0.0739
IYR                    0.0818                   0.1685        0.1283
PSP                    0.0995                   0.2134        0.1252
SPY                    0.1350                   0.1428        0.2591
TIP                    0.0274                   0.0513        0.1153
```

### 2.2 Descriptive Analysis

#### Correlation Matrix
```
       BWX     DBC    EEM    EFA    HYG     IEF    IYR    PSP    SPY    TIP
BWX 1.0000  0.1882 0.6221 0.6040 0.6076  0.5895 0.5508 0.5255 0.4408 0.6837
DBC 0.1882  1.0000 0.5109 0.4995 0.4563 -0.3022 0.2776 0.4523 0.4297 0.0998
EEM 0.6221  0.5109 1.0000 0.8204 0.6927  0.0336 0.5843 0.7502 0.6884 0.3821
EFA 0.6040  0.4995 0.8204 1.0000 0.7884  0.0498 0.6992 0.8951 0.8461 0.3980
HYG 0.6076  0.4563 0.6927 0.7884 1.0000  0.1958 0.7376 0.8114 0.7927 0.5408
IEF 0.5895 -0.3022 0.0336 0.0498 0.1958  1.0000 0.3152 0.0254 0.0047 0.7595
IYR 0.5508  0.2776 0.5843 0.6992 0.7376  0.3152 1.0000 0.7498 0.7543 0.5951
PSP 0.5255  0.4523 0.7502 0.8951 0.8114  0.0254 0.7498 1.0000 0.8918 0.4073
SPY 0.4408  0.4297 0.6884 0.8461 0.7927  0.0047 0.7543 0.8918 1.0000 0.3806
TIP 0.6837  0.0998 0.3821 0.3980 0.5408  0.7595 0.5951 0.4073 0.3806 1.0000
```

**Key Findings:**
- Highest correlation: ('EFA', 'PSP') = 0.8951
- Lowest correlation: ('DBC', 'IEF') = -0.3022

### 2.3 Mean-Variance Frontier Analysis

#### Tangency Portfolio (Full)
```
BWX   -0.8654
DBC   -0.0680
EEM    0.0258
EFA    0.0649
HYG    0.2891
IEF    0.8744
IYR   -0.2427
PSP   -0.3334
SPY    1.0632
TIP    0.1922
```
- Annualized return: 0.1361
- Annualized volatility: 0.0879
- Sharpe ratio: 0.4244

#### 2.4 TIPS Analysis

##### Tangency Portfolio without TIPS
```
BWX   -0.8975
DBC   -0.0594
EEM    0.0324
EFA    0.0556
HYG    0.3171
IEF    1.0268
IYR   -0.2437
PSP   -0.3406
SPY    1.1093
```
- Annualized return without TIPS: 0.1409
- Annualized volatility without TIPS: 0.0912
- Sharpe ratio without TIPS: 0.4238

##### Tangency Portfolio with TIPS Adjusted
```
BWX   -0.6809
DBC   -0.1170
EEM   -0.0127
EFA    0.1187
HYG    0.1273
IEF   -0.0044
IYR   -0.2369
PSP   -0.2920
SPY    0.7977
TIP    1.3002
```
- Annualized return with TIPS adjusted: 0.1277
- Annualized volatility with TIPS adjusted: 0.0748
- Sharpe ratio with TIPS adjusted: 0.4666

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

#### Scaled EW weights (target μ = 0.01):
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
- Annualized Return: 0.1383
- Annualized Volatility: 0.2695
- Sharpe Ratio: 0.1286
- Monthly Excess Return: 0.0100

### 3-2. Risk-parity (RP) portfolio

#### Risk-parity weights (inverse variance):
```
BWX   0.1238
DBC   0.0314
EEM   0.0278
EFA   0.0379
HYG   0.1494
IEF   0.2102
IYR   0.0304
PSP   0.0190
SPY   0.0424
TIP   0.3278
```

#### Scaled RP weights (target μ = 0.01):
```
BWX   0.5332
DBC   0.1351
EEM   0.1197
EFA   0.1631
HYG   0.6438
IEF   0.9056
IYR   0.1311
PSP   0.0818
SPY   0.1826
TIP   1.4124
```

**Performance:**
- Annualized Return: 0.1497
- Annualized Volatility: 0.2646
- Sharpe Ratio: 0.1309
- Monthly Excess Return: 0.0100

### 3-3. Mean-Variance (MV) portfolio

#### Original MV (tangency) weights:
```
BWX   -0.8654
DBC   -0.0680
EEM    0.0258
EFA    0.0649
HYG    0.2891
IEF    0.8744
IYR   -0.2427
PSP   -0.3334
SPY    1.0632
TIP    0.1922
```

#### Scaled MV weights (target μ = 0.01):
```
BWX   -0.8035
DBC   -0.0631
EEM    0.0239
EFA    0.0603
HYG    0.2684
IEF    0.8119
IYR   -0.2254
PSP   -0.3096
SPY    0.9872
TIP    0.1784
```

**Performance:**
- Annualized Return: 0.1264
- Annualized Volatility: 0.0816
- Sharpe Ratio: 0.4244
- Monthly Excess Return: 0.0100

### 3-4. Performance Comparison

```
                       Equally-Weighted  Risk-Parity  Mean-Variance
Annualized Return                0.1383       0.1497         0.1264
Annualized Volatility            0.2695       0.2646         0.0816
Sharpe Ratio                     0.1286       0.1309         0.4244
```

**Best Sharpe Ratio:** Mean-Variance (0.4244)  
**Lowest Volatility:** Mean-Variance (0.0816)

### Target Monthly Excess Return Verification:
- EW: 0.010000
- RP: 0.010000
- MV: 0.010000
- Target: 0.010000

## Key Insights

- **Mean-Variance optimization** provides the best risk-adjusted returns (Sharpe ratio: 0.4244)
- **Mean-Variance** provides the lowest volatility (0.0816 vs EW: 0.2695, RP: 0.2646)
- **Risk-Parity** achieves the highest return (0.1497) but with much higher volatility
- All three methods successfully achieve the target monthly excess return of 0.0100
- **Mean-Variance dominates** both alternatives in risk-adjusted performance
