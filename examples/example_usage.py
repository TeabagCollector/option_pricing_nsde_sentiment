"""
期权数据获取使用示例
需要设置环境变量 RQDATAC_LICENSE（米筐 API）
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # project root

from OptionDataFetcher import OptionDataFetcher
import pandas as pd

# =============================================================================
# 示例 1: 获取单个合约的完整数据
# =============================================================================
print("="*60)
print("示例 1: 获取单个合约数据")
print("="*60)

fetcher = OptionDataFetcher()
fetcher.init_connection()  # 从 RQDATAC_LICENSE 环境变量读取

# 获取一个看涨期权
df_call = fetcher.get_option_data(
    order_book_id='IO2412C3850',
    start_date='20241201',
    end_date='20241220',
    include_greeks=True
)

print(f"\n获取到 {len(df_call)} 条记录")
print(f"日期范围: {df_call.index.get_level_values('date').min()} 至 {df_call.index.get_level_values('date').max()}")

# 查看期权定价所需的核心字段
core_fields = ['strike_price', 'close', 'time_to_expire', 'risk_free_rate', 
               'underlying_close', 'hv_20d', 'iv', 'call_put', 'euro_american']
print(f"\n核心字段数据预览:")
print(df_call[core_fields].head())

# 保存数据
fetcher.save_data(df_call, 'example_single_contract.csv')
print(f"\n✓ 数据已保存到 example_single_contract.csv")


# =============================================================================
# 示例 2: 批量获取多个合约（看涨+看跌）
# =============================================================================
print("\n" + "="*60)
print("示例 2: 批量获取看涨和看跌期权")
print("="*60)

contracts = [
    'IO2412C3850',  # 看涨
    'IO2412P3850',  # 看跌
    'IO2412C4000',  # 看涨，不同行权价
]

df_multi = fetcher.get_option_data(
    order_book_id=contracts,
    start_date='20241201',
    end_date='20241220'
)

print(f"\n获取到 {len(df_multi)} 条记录")
print(f"涵盖合约: {df_multi.index.get_level_values('order_book_id').unique().tolist()}")

# 按合约分组统计
print("\n各合约数据量:")
print(df_multi.groupby(level='order_book_id').size())

fetcher.save_data(df_multi, 'example_multiple_contracts.csv')
print(f"\n✓ 数据已保存到 example_multiple_contracts.csv")


# =============================================================================
# 示例 3: 提取模型训练所需的特征
# =============================================================================
print("\n" + "="*60)
print("示例 3: 准备神经网络训练数据")
print("="*60)

# 选择模型输入特征（参考Version_5的特征顺序）
feature_columns = ['underlying_close', 'strike_price', 'risk_free_rate', 
                   'time_to_expire', 'hv_20d', 'call_put', 'euro_american']
target_column = 'close'  # 期权价格作为目标变量

# 提取特征矩阵和目标变量
X = df_multi[feature_columns].values
y = df_multi[target_column].values

print(f"\n特征矩阵 X 形状: {X.shape}")
print(f"目标变量 y 形状: {y.shape}")
print(f"\n特征列: {feature_columns}")

# 查看特征统计信息
print(f"\n特征统计信息:")
print(df_multi[feature_columns].describe())


# =============================================================================
# 示例 4: 数据质量检查
# =============================================================================
print("\n" + "="*60)
print("示例 4: 数据质量检查")
print("="*60)

# 检查缺失值
print(f"\n缺失值统计:")
missing = df_multi[core_fields].isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "无缺失值")

# 检查异常值
print(f"\n期权价格分布:")
print(df_multi['close'].describe())

print(f"\n实值程度(moneyness)分布:")
print(df_multi['moneyness'].describe())

# Greeks统计
print(f"\nDelta分布:")
print(df_multi['delta'].describe())

print(f"\n隐含波动率(IV)分布:")
print(df_multi['iv'].describe())


# =============================================================================
# 示例 5: 按日期查看某一天的期权数据
# =============================================================================
print("\n" + "="*60)
print("示例 5: 查看特定日期的期权数据")
print("="*60)

# 选择一个日期
target_date = '2024-12-19'
df_date = df_multi.loc[(slice(None), target_date), :]

print(f"\n{target_date} 的期权数据:")
print(df_date[['strike_price', 'close', 'underlying_close', 'iv', 'delta', 'call_put']])


# =============================================================================
# 示例 6: 获取指定月份到期的合约列表
# =============================================================================
print("\n" + "="*60)
print("示例 6: 查询2024年各月到期的期权合约")
print("="*60)

# 查询2024年各月到期的合约
monthly_contracts = fetcher.get_monthly_contracts(
    year=2024,
    underlying='000300.XSHG'
)

# 查看部分结果
for month_str, contracts in list(monthly_contracts.items())[:3]:
    print(f"\n{month_str} 到期的合约（前5个）:")
    print(contracts[:5])


# =============================================================================
# 示例 7: 获取全年期权数据（批量获取）
# =============================================================================
print("\n" + "="*60)
print("示例 7: 批量获取2024年全年期权数据")
print("="*60)

# 注意：这个操作可能需要较长时间，建议在实际使用时取消注释
# df_annual = fetcher.get_annual_data(
#     year=2024,
#     underlying='000300.XSHG',
#     save_monthly=True,  # 按月保存，避免单文件过大
#     include_greeks=True
# )
# 
# print(f"\n年度数据统计:")
# print(f"  总记录数: {len(df_annual)}")
# print(f"  合约数量: {df_annual.index.get_level_values('order_book_id').nunique()}")
# print(f"  日期范围: {df_annual.index.get_level_values('date').min()} ~ {df_annual.index.get_level_values('date').max()}")

print("\n提示: 要获取全年数据，请取消注释上方代码")
print("注意: 全年数据获取可能需要10-30分钟，数据会自动按月保存到 data/annual/ 目录")


# =============================================================================
# 示例 8: 查询合约的交易日期范围
# =============================================================================
print("\n" + "="*60)
print("示例 8: 查询合约的实际交易日期")
print("="*60)

contract_id = 'IO2412C3850'
listed_date, maturity_date = fetcher.get_contract_trading_dates(contract_id)

print(f"\n合约 {contract_id}:")
print(f"  上市日期: {listed_date}")
print(f"  到期日期: {maturity_date}")


print(f"\n完成！所有示例运行完毕。")
