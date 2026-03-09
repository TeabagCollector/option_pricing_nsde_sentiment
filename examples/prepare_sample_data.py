"""
数据采样脚本 - 生成固定的训练集和测试集样本

功能：
1. 读取完整的训练集和测试集
2. 使用固定的 random_state=42 进行采样
3. 导出到 data/sample/ 目录
4. 确保每次实验使用相同的数据

Author: Version_9
Date: 2026-02-21
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # project root

import pandas as pd
import numpy as np
from pathlib import Path
from OptionPricingModel import extract_expiry_yymm

# 配置参数
TRAIN_SAMPLE_SIZE = 2000
TEST_SAMPLE_SIZE = 500
RANDOM_STATE = 42

def clean_option_data(d):
    """数据清洗函数"""
    d = d.dropna(subset=["underlying_close", "strike_price", "time_to_expire", 
                          "risk_free_rate", "close", "hv_20d", "call_put", "iv"])
    d = d[d["time_to_expire"] > 5/365]  # 剔除到期时间小于5天的"末日轮"
    d = d[d["close"] >= 0.2]  # 期权最低结算价 0.2
    d = d[d["hv_20d"] > 1e-6]
    d = d[d["iv"] > 1e-8]  # 排除 iv 缺失或异常小值
    return d

def main():
    print("="*60)
    print("数据采样脚本")
    print("="*60)
    
    # 读取完整数据
    print("\n1. 读取完整数据...")
    df = pd.read_csv("full_option_trading_data.csv")
    df["expiry_yymm"] = df["order_book_id"].apply(extract_expiry_yymm)
    
    # 划分训练集和测试集
    train_yymm = [f"24{i:02d}" for i in range(1, 12)]
    df_train = df[df["expiry_yymm"].isin(train_yymm)].copy()
    df_test = df[df["expiry_yymm"] == "2412"].copy()
    
    # 清洗数据
    print("2. 清洗数据...")
    df_train = clean_option_data(df_train)
    df_test = clean_option_data(df_test)
    
    print(f"   训练集大小: {len(df_train)}")
    print(f"   测试集大小: {len(df_test)}")
    
    # 采样
    print(f"\n3. 采样数据 (random_state={RANDOM_STATE})...")
    df_train_sample = df_train.sample(n=TRAIN_SAMPLE_SIZE, random_state=RANDOM_STATE)
    df_test_sample = df_test.sample(n=TEST_SAMPLE_SIZE, random_state=RANDOM_STATE)
    
    print(f"   训练集采样: {len(df_train)} → {len(df_train_sample)}")
    print(f"   测试集采样: {len(df_test)} → {len(df_test_sample)}")
    
    # 创建输出目录
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导出CSV
    train_file = output_dir / f"train_{TRAIN_SAMPLE_SIZE}_seed{RANDOM_STATE}.csv"
    test_file = output_dir / f"test_{TEST_SAMPLE_SIZE}_seed{RANDOM_STATE}.csv"
    
    print(f"\n4. 导出文件...")
    df_train_sample.to_csv(train_file, index=False)
    df_test_sample.to_csv(test_file, index=False)
    
    print(f"   ✓ 训练集: {train_file}")
    print(f"   ✓ 测试集: {test_file}")
    
    # 验证信息
    print(f"\n5. 验证信息:")
    print(f"   训练集:")
    print(f"     - Call期权: {(df_train_sample['call_put'] == 0).sum()}")
    print(f"     - Put期权:  {(df_train_sample['call_put'] == 1).sum()}")
    print(f"     - 价格范围: [{df_train_sample['close'].min():.2f}, {df_train_sample['close'].max():.2f}]")
    print(f"   测试集:")
    print(f"     - Call期权: {(df_test_sample['call_put'] == 0).sum()}")
    print(f"     - Put期权:  {(df_test_sample['call_put'] == 1).sum()}")
    print(f"     - 价格范围: [{df_test_sample['close'].min():.2f}, {df_test_sample['close'].max():.2f}]")
    
    print("\n" + "="*60)
    print("✅ 采样数据集生成完成！")
    print("="*60)

if __name__ == "__main__":
    main()
