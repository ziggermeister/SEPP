# SEPP Framework

This repository contains the **SEPP Composite Score Framework (v6.4.3)** with validation harness and real-time data support.

## Structure
- `sepp_engine.py`: Core simulation engine
- `validation_harness_pack.py`: Validation harness with regression, golden master, and metamorphic tests
- `params_ingest.py`: Helper to load parameters
- `cli.py`: Command-line entry point
- `adapters/`: Data providers (Yahoo Finance, Tiingo, Alpha Vantage, etc.)
- `ingest/`: Data ingestion and feature engineering
- `portfolio/`: Portfolio schema, backtesting, visualization
- `data/`: Symbol metadata and cached data

## Usage
```bash
python cli.py --params params.yaml
```

## Installation
```bash
pip install -r requirements.txt
```

## Features Roadmap
- Real-time data ingestion from multiple providers
- Portfolio evaluation on arbitrary historical start dates
- Visual portfolio comparisons and growth projections
- SEPP withdrawal modeling and stress testing
