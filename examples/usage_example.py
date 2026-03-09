"""
损失函数与初始方差优化 - 使用示例

演示如何使用新增的功能：
1. MAPE 损失函数
2. ATM 隐含波动率作为初始方差
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # project root

import pandas as pd
import numpy as np
from NeuralSDEPricer import NeuralSDEPricer
from OptionPricingModel import compute_daily_atm_iv

# 示例1：使用默认参数（向后兼容）
print("=" * 60)
print("示例1：默认参数（MSE损失 + hv_20d）")
print("=" * 60)

model_default = NeuralSDEPricer(
    latent_dim=2,
    n_paths=5000,
    n_steps=50
)

print(f"损失函数类型: {model_default.loss_type}")
print(f"初始方差来源: {model_default.v0_source}")
print()

# 示例2：使用MAPE损失函数
print("=" * 60)
print("示例2：MAPE损失 + hv_20d（推荐用于虚值期权较多的场景）")
print("=" * 60)

model_mape = NeuralSDEPricer(
    latent_dim=2,
    loss_type='mape',       # 使用MAPE损失
    v0_source='hv_20d',     # 保持使用历史波动率
    n_paths=5000
)

print(f"损失函数类型: {model_mape.loss_type}")
print(f"初始方差来源: {model_mape.v0_source}")
print("\n优势：对所有价格区间的期权一视同仁，不会偏重高价期权")
print()

# 示例3：使用ATM隐含波动率
print("=" * 60)
print("示例3：MSE损失 + ATM IV（推荐用于提升前瞻性）")
print("=" * 60)

model_atm_iv = NeuralSDEPricer(
    latent_dim=2,
    loss_type='mse',
    v0_source='atm_iv',     # 使用每日ATM隐含波动率
    n_paths=5000
)

print(f"损失函数类型: {model_atm_iv.loss_type}")
print(f"初始方差来源: {model_atm_iv.v0_source}")
print("\n优势：市场前瞻性，无数据泄露（使用当日其他期权的IV计算）")
print()

# 示例4：最佳组合（推荐）
print("=" * 60)
print("示例4：MAPE损失 + ATM IV（推荐组合）")
print("=" * 60)

model_best = NeuralSDEPricer(
    latent_dim=2,
    loss_type='mape',       # 对所有价格区间公平
    v0_source='atm_iv',     # 市场前瞻性
    n_paths=5000,
    n_steps=50,
    random_state=42
)

print(f"损失函数类型: {model_best.loss_type}")
print(f"初始方差来源: {model_best.v0_source}")
print("\n优势：")
print("  - MAPE损失：虚值期权和实值期权得到同等重视")
print("  - ATM IV：使用市场共识的前瞻性波动率，比历史波动率更准确")
print()

# 示例5：训练时动态指定损失函数
print("=" * 60)
print("示例5：训练时覆盖默认损失函数")
print("=" * 60)

model_override = NeuralSDEPricer(
    latent_dim=2,
    loss_type='mse',        # 默认使用MSE
    v0_source='hv_20d'
)

print(f"模型默认损失函数: {model_override.loss_type}")
print("\n可以在训练时覆盖：")
print("  model.fit(df_train, sentiment_dict, loss_type='mape')")
print("  # 这次训练会使用 MAPE 而不是 MSE")
print()

# 示例6：计算每日ATM隐含波动率（数据预处理）
print("=" * 60)
print("示例6：独立使用 compute_daily_atm_iv")
print("=" * 60)

# 创建示例数据
dates = pd.date_range('2024-01-01', periods=3)
df_example = pd.DataFrame({
    'date': np.repeat(dates, 5),
    'underlying_close': 3500.0,
    'strike_price': [3400, 3450, 3500, 3550, 3600] * 3,
    'iv': np.random.uniform(0.15, 0.25, 15)
})

daily_atm_iv = compute_daily_atm_iv(df_example)

print("计算每日ATM隐含波动率：")
print(daily_atm_iv)
print(f"\nATM IV 统计：")
print(f"  最小值: {daily_atm_iv.min():.4f}")
print(f"  最大值: {daily_atm_iv.max():.4f}")
print(f"  平均值: {daily_atm_iv.mean():.4f}")
print()

# 示例7：三种损失函数对比
print("=" * 60)
print("示例7：三种损失函数特性对比")
print("=" * 60)

import torch

pred = torch.tensor([5.0, 50.0, 200.0])    # 低价、中价、高价期权
target = torch.tensor([6.0, 45.0, 180.0])

loss_mse = NeuralSDEPricer._compute_loss(pred, target, 'mse')
loss_mape = NeuralSDEPricer._compute_loss(pred, target, 'mape')
loss_rel_mse = NeuralSDEPricer._compute_loss(pred, target, 'relative_mse')

print("预测值: [5.0, 50.0, 200.0]")
print("真实值: [6.0, 45.0, 180.0]")
print("\n损失函数值：")
print(f"  MSE:          {loss_mse.item():.2f}  (被高价期权主导)")
print(f"  MAPE:         {loss_mape.item():.2f}  (所有期权平等)")
print(f"  Relative MSE: {loss_rel_mse.item():.2f}  (介于两者之间)")
print()

print("=" * 60)
print("总结")
print("=" * 60)
print("\n推荐配置：")
print("  1. 数据集虚值期权较多 → loss_type='mape'")
print("  2. 追求市场前瞻性 → v0_source='atm_iv'")
print("  3. 最佳实践 → loss_type='mape' + v0_source='atm_iv'")
print("  4. 向后兼容 → 不传参数，自动使用 loss_type='mse' + v0_source='hv_20d'")
