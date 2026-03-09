# Neural SDE & Sentiment-Integrated Option Pricing

Hybrid Heston-Neural SDE framework for CSI 300 index options, with sentiment latent factor learned from Risk Reversal surfaces via VAE. Following Wang & Hong (2021).

## Features

- **Risk Reversal Surface VAE**: Learn 2D sentiment latent $z$ from RR surfaces; implicit neural representation
- **Hybrid Heston-Neural SDE**: 4 NNs parameterizing drift/diffusion; sentiment $z$ injected into dynamics
- **Monte Carlo pricing**: Antithetic variates; validated on CSI 300 index options
- **OptionDataFetcher**: Fetch option data & Greeks via RiceQuant (Miqiang) API

## Project Structure

```
├── OptionDataFetcher.py      # Data fetching (RiceQuant)
├── RiskReversalSurfaceVAE.py # VAE for RR surface → latent z
├── NeuralSDEPricer.py        # Sentiment-integrated NSDE
├── NeuralSDEPricerNoSentiment.py
├── OptionPricingModel.py      # Heston baseline, metrics
├── FlexibleMLP.py
├── OptionPricingReporter.py
├── VolatilitySurfaceVisualizer.py
├── Notebooks/                 # Pipeline notebooks
│   ├── stage_0_playground.ipynb
│   ├── stage_1_vae_for_risk_reversal_sruface.ipynb
│   ├── stage_2_heston.ipynb
│   ├── stage_3_neural_networks.ipynb
│   ├── stage_4_sentiment_integrated.ipynb
│   ├── stage_5_nsde.ipynb
│   └── stage_5.1_nsde_no_sentiment.ipynb
├── examples/                 # Usage examples
│   ├── example_usage.py      # OptionDataFetcher
│   ├── usage_example.py      # NeuralSDEPricer (MAPE, ATM IV)
│   ├── example_visualizer_usage.py
│   ├── prepare_sample_data.py
│   └── test_contract.py
├── data/
│   ├── sample/               # Sample data for quick run
│   ├── train/                # Full train data (gitignored)
│   └── test/                 # Full test data (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
pip install -r requirements.txt

# RiceQuant API (for OptionDataFetcher)
pip install rqdatac
export RQDATAC_LICENSE='your_license_key'
# Or copy .env.example to .env and fill in
```

## Usage

**Run from project root** so imports resolve correctly.

```python
from OptionDataFetcher import OptionDataFetcher
from NeuralSDEPricer import NeuralSDEPricer

fetcher = OptionDataFetcher()
fetcher.init_connection()  # reads RQDATAC_LICENSE from env

model = NeuralSDEPricer(latent_dim=2, n_paths=5000, loss_type='mape')
# ... see examples/usage_example.py
```

Or run notebooks in `Notebooks/` for the full pipeline (stage 0 → 5).

## Data

- **CSI 300 options**: Requires RiceQuant license; fetched via `OptionDataFetcher`
- **Sample data**: `data/sample/` (train_2000_seed42.csv, test_500_seed42.csv) for quick runs without API

## References

- Wang & Hong (2021). Neural SDE for option pricing.
- Gatheral et al. (2014). Volatility is rough.

## License

MIT
