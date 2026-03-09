# Neural SDE & Sentiment-Integrated Option Pricing

Hybrid Heston-Neural SDE framework for CSI 300 index options, with sentiment latent factor learned from Risk Reversal surfaces via VAE. Following Wang & Hong (2021).

## Features

- **Risk Reversal Surface VAE**: Learn 2D sentiment latent $z$ from RR surfaces; implicit neural representation
- **Hybrid Heston-Neural SDE**: 4 NNs parameterizing drift/diffusion; sentiment $z$ injected into dynamics
- **Monte Carlo pricing**: Antithetic variates; validated on CSI 300 index options

## Pipeline (Stage 1 → 5)

| Stage | Notebook | 说明 |
|-------|----------|------|
| 1 | `stage_1_vae_for_risk_reversal_sruface.ipynb` | RR Surface VAE 训练，输出 latent z |
| 2 | `stage_2_heston.ipynb` | BS / Heston 基准 |
| 3 | `stage_3_neural_networks.ipynb` | NN(MSE) / NN(自定义损失) baseline |
| 4 | `stage_4_sentiment_integrated.ipynb` | NN + Sentiment 整合 |
| 5 | `stage_5_nsde.ipynb` | NSDE + Sentiment 完整模型 |
| 5.1 | `stage_5.1_nsde_no_sentiment.ipynb` | NSDE 无 Sentiment 对照 |

## Project Structure

```
├── RiskReversalSurfaceVAE.py # VAE for RR surface → latent z
├── NeuralSDEPricer.py        # Sentiment-integrated NSDE
├── NeuralSDEPricerNoSentiment.py
├── OptionPricingModel.py     # Heston baseline, metrics
├── FlexibleMLP.py
├── OptionPricingReporter.py
├── VolatilitySurfaceVisualizer.py
├── Notebooks/                 # Pipeline (stage 1–5)
├── examples/
│   ├── usage_example.py      # NeuralSDEPricer (MAPE, ATM IV)
│   ├── example_visualizer_usage.py
│   └── prepare_sample_data.py
├── data/
│   └── sample/               # 小样本 (train_2000, test_500)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data（无需 API）

将 `full_option_trading_data.csv` 放在项目根目录即可运行 pipeline。该文件格式：CSI 300 期权 2024 年数据，含 `underlying_close`, `strike_price`, `time_to_expire`, `risk_free_rate`, `close`, `hv_20d`, `call_put`, `iv` 等列。

- 若已有该文件：直接运行 stage 1 → 5
- 若无：可用 `examples/prepare_sample_data.py` 从 `data/train/` + `data/test/` 合并生成（需先有原始数据）

## Key Results（CSI 300, 测试集 2412）

| 模型 | MAE | RMSE | MAPE(%) |
|------|-----|------|---------|
| Heston | — | — | 基准 |
| NN(MSE) | ~18.8 | ~31.8 | ~38 |
| NN(自定义) | ~20.2 | ~35.2 | ~24 |
| NN+Sentiment | ~20.6 | ~32.6 | ~48 (MSE) / ~24 (自定义) |
| NSDE+Sentiment | 见 stage_5 输出 | | |

Sentiment 在 NN(自定义) 上 MAPE 持平；NSDE 需完整训练后查看 metrics。

## Usage

```python
from NeuralSDEPricer import NeuralSDEPricer
from OptionPricingModel import compute_metrics

model = NeuralSDEPricer(latent_dim=2, n_paths=5000, loss_type='mape')
# 见 examples/usage_example.py
```

## References

- Wang & Hong (2021). Neural SDE for option pricing.
- Gatheral et al. (2014). Volatility is rough.

## License

MIT
