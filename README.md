# Structure-Aware Financial Forecasting using Hankel Matrices and Dual-LSTM Fusion

A structure-aware financial forecasting framework that combines **Hankel matrix decomposition**, **Singular Value Decomposition (SVD)**, **dual-branch LSTM learning**, and **Monte Carlo uncertainty estimation** for multi-asset financial analysis.

---

## Overview

Financial markets are influenced by interactions between multiple asset classes rather than isolated price movements. Traditional forecasting models often operate on a single asset stream and fail to capture structural relationships that emerge during periods of market stress.

This project introduces a forecasting pipeline that models the **structural state of financial markets** instead of directly predicting raw prices.

The framework integrates:

* Multi-asset financial data
* Micro-level technical indicators
* Macro-level cross-market relationships
* Hankel matrix structural embeddings
* Singular Value Decomposition (SVD)
* Dual-branch LSTM fusion
* Monte Carlo Dropout uncertainty estimation

The model was evaluated using historical data from:

* Tata Consultancy Services (TCS)
* Bitcoin (BTC/USD)
* Crude Oil WTI Futures
* USD/INR Exchange Rate

covering the period from **January 2020 to April 2026**.

---

## Research Motivation

During major market disruptions such as the COVID-19 crash, commodity shocks, and cryptocurrency collapses, assets frequently move together despite belonging to different market domains.

A forecasting model trained on a single asset cannot directly observe these interactions.

This project explores whether structural properties extracted from multi-asset windows using Hankel matrices and SVD can provide more informative forecasting signals than raw financial features alone.

---

## Methodology

### 1. Data Collection & Preprocessing

Historical OHLCV data are collected for:

* TCS Equity
* BTC/USD
* Crude Oil WTI Futures
* USD/INR Exchange Rate

The datasets are:

* cleaned
* aligned temporally
* normalized
* merged into a unified multi-asset dataset

---

### 2. Feature Engineering

#### Micro Features

Asset-specific indicators:

* Returns
* Log Returns
* Daily Range
* Candle Body
* Moving Averages
* Volatility
* Momentum

#### Macro Features

Cross-market indicators:

* Rolling Correlations
* Cross-Asset Ratios
* Market Stress Index
* Relative Return Spreads

---

### 3. Structural Modeling

Instead of feeding raw features directly into the neural network:

1. Sliding windows are converted into Hankel matrices.
2. Singular Value Decomposition is applied.
3. Three structural descriptors are extracted:

* Dominance Ratio
* Spectral Entropy
* Effective Rank

These descriptors form a compact representation of market structure.

---

### 4. Dual-LSTM Fusion Network

Two separate LSTM branches process:

* Micro structural sequences
* Macro structural sequences

The representations are combined through:

* Temporal Attention
* Gated Fusion
* Dense Prediction Head

to forecast the next structural state.

---

### 5. Uncertainty Estimation

Monte Carlo Dropout is applied during inference.

Multiple stochastic forward passes generate:

* Mean Forecast
* Predictive Uncertainty
* Confidence Bands

allowing the model to express uncertainty rather than providing only point predictions.

---

## Repository Structure

```text
Structure-Aware-Financial-Forecasting/
│
├── README.md
│
├── code/
│   └── data/
│   |   ├── btc_usd_binance_historical_data.csv
│   |   ├── crude_oil_wti_futures_historical_data.csv
│   |   ├── tata_consultancy_stock_price_history.csv
│   |   └── usd_inr_historical_data.csv
|   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── hankelization.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   ├── monte_carlo.py
│
└── documentation/
    ├── research_paper.pdf
    └── presentation.pptx
```

---

## Experimental Configuration

| Parameter        | Value            |
| ---------------- | ---------------- |
| Observations     | 1553             |
| Train/Test Split | 80/20            |
| Micro Features   | 55               |
| Macro Features   | 8                |
| Time Window      | 20               |
| Hankel Window    | 8                |
| Sequence Length  | 30               |
| Micro LSTM       | 64 Hidden Units  |
| Macro LSTM       | 128 Hidden Units |
| Optimizer        | Adam             |
| Learning Rate    | 5e-4             |
| Batch Size       | 32               |
| Epochs           | 20               |
| Dropout          | 0.3              |
| MC Samples       | 50               |

---

## Results

| Metric         | Score  |
| -------------- | ------ |
| MSE            | 0.0306 |
| RMSE           | 0.1749 |
| Trend Accuracy | 66.44% |

### Key Observation

The strongest result of the current implementation is directional forecasting performance.

While the structural target construction introduces limitations that affect absolute regression accuracy, the model achieves a trend accuracy of **66.44%**, suggesting that meaningful directional information is being extracted from the structural score sequences.

---

## Limitations

This repository intentionally documents its current limitations:

* Structural target variance collapse due to effective-rank dominance
* No hyperparameter ablation study
* No vanilla LSTM baseline on identical targets
* Uncalibrated Monte Carlo uncertainty estimates
* Limited exploration of adaptive Hankel window sizes

These limitations provide clear directions for future work.

---

## Future Work

* Component-wise structural score prediction
* Attention-based fusion mechanisms
* Adaptive Hankel embeddings
* Change-point and regime detection
* Uncertainty calibration
* Real-time forecasting deployment
* Explainable AI integration

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python eval.py
```

---

## Monte Carlo Uncertainty Analysis

```bash
python monte_carlo.py
```

---

## Authors

Dhwanit Sharma

Kanan Goel

Parkhi Mahajan

Punit Lohan

Department of Artificial Intelligence and Data Science

Amrita Vishwa Vidyapeetham

Faridabad, India

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{sharma2025structureaware,
  title={Structure-Aware Financial Forecasting using Hankel Matrices and Dual LSTM Fusion},
  author={Dhwanit Sharma and Kanan Goel and Parkhi Mahajan and Punit Lohan},
  year={2025},
  institution={Amrita Vishwa Vidyapeetham}
}
```
