# Macro_Valuation_Engine
# Automated Multi-Asset Valuation & Macro Sentiment Backtesting Engine

An institutional-grade quantitative financial engineering tool that pairs fundamental corporate valuation with real-time natural language processing (NLP). The engine ingests real-time financial statements, extracts key balance sheet aggregates, applies transformer-based sentiment analysis to macro news regimes, and dynamically computes intrinsic values across an equity universe.

## Core Architecture & Features

* **Dynamic Capital Structure Modeling:** Implements a top-down Free Cash Flow to Equity (FCFE) multi-period forecasting engine utilizing real-time capital data via `yfinance`.
* **Transformer-Based Sentiment Filter:** Integrates a local Hugging Face NLP pipeline (`DistilBERT`) to parse live market text and financial news headlines, mapping qualitative macro regimes into quantitative sentiment scores between $-1.0$ and $+1.0$.
* **Adaptive Terminal Growth Rates:** Replaces rigid, static historical projections with a responsive algorithm that scales terminal growth rates dynamically based on the NLP sentiment vectors.
* **Cross-Sectional Screening & Backtesting:** Compiles, ranks, and screens an equity basket based on relative margins of safety, benchmarking systematic performance against the S&P 500 (`^GSPC`) index.

---

## System Requirements & Local Setup

### Dependencies
Ensure you have Python 3.8+ installed along with the required analytical and machine learning stack:
```bash
pip install numpy pandas yfinance requests beautifulsoup4 transformers torch
