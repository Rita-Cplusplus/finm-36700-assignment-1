# FINM 36700 Assignment 1 - Mean-Variance Optimization

This repository contains the implementation of Mean-Variance Portfolio Optimization analysis for FINM 36700.

## Project Structure

```
├── assignment1.py              # Problems 2-3: Summary Statistics and Allocations
├── assignment_extra.py         # Problems 4-7: Out-of-Sample, Bayesian, and Extended Analysis
├── multi_asset_etf_data.xlsx   # Data file with ETF returns
├── problems_2_3_output.md      # Output results for Problems 2-3
├── problems_4_7_output.md      # Output results for Problems 4-7
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rita-Cplusplus/finm-36700-assignment-1.git
cd finm-36700-assignment-1
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Problems 2-3 (Summary Statistics and Allocations):
```bash
python assignment1.py
```

### Run Problems 4-7 (Out-of-Sample and Extended Analysis):
```bash
python assignment_extra.py
```

## Analysis Overview

### Problems 2-3 (`assignment1.py`)
- **2. Summary Statistics and Descriptive Analysis**
  - 2.1 Annualized Statistics
  - 2.2 Descriptive Analysis (Correlation Matrix)
  - 2.3 Mean-Variance Frontier Analysis
  - 2.4 TIPS Analysis
- **3. Allocations** (Target monthly excess return μ = 0.01)
  - 3-1. Equally-weighted (EW) portfolio
  - 3-2. Risk-parity (RP) portfolio
  - 3-3. Mean-Variance (MV) portfolio
  - 3-4. Performance Comparison

### Problems 4-7 (`assignment_extra.py`)
- **4. Out-of-Sample Performance**
  - 4.1 One-step Out-of-Sample Performance
  - 4.2 Rolling Out-of-Sample Performance
- **5. EXTRA: Without a Riskless Asset**
  - 5.1 Minimum Variance Portfolio
  - 5.2 Mean-Variance Frontier without Risk-free Asset
- **6. EXTRA: Bayesian Allocation**
  - 6.1 Regularized Allocation Implementation
  - 6.2 Bayesian Shrinkage Analysis
- **7. EXTRA: Inefficient Tangency**
  - 7.1 Analysis with QAI included
  - 7.2 Impact Analysis

## Data

The analysis uses `multi_asset_etf_data.xlsx` containing:
- **Assets**: BWX, DBC, EEM, EFA, HYG, IEF, IYR, PSP, SPY, TIP (and QAI for Problem 7)
- **Sheets**: 
  - `total returns`: Total return data
  - `excess returns`: Excess return data over risk-free rate

## Key Results

- **Best Sharpe Ratio**: Mean-Variance portfolio (0.4244)
- **Lowest Volatility**: Mean-Variance portfolio (0.0816)
- **Highest Return**: Risk-Parity portfolio (0.1497)
- All portfolio methods successfully achieve the target monthly excess return of 0.0100

## Dependencies

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.24.0
- openpyxl >= 3.0.0
- scipy >= 1.9.0

## Output Files

Detailed results are saved in:
- `problems_2_3_output.md`: Complete output for Problems 2-3
- `problems_4_7_output.md`: Complete output for Problems 4-7