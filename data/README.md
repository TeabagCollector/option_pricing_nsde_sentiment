# Data Directory

- **sample/**: 小样本 (train_2000_seed42.csv, test_500_seed42.csv)，可提交
- **train/**, **test/**: 完整数据 (gitignored)

## 仅用已有数据运行（无 API）

1. 将 `full_option_trading_data.csv` 放在**项目根目录**
2. 直接运行 Notebooks stage 1 → 5
3. 该文件格式：CSI 300 期权 2024 年，含 order_book_id, date, underlying_close, strike_price, time_to_expire, risk_free_rate, close, hv_20d, call_put, iv 等
