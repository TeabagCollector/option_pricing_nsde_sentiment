"""
测试米筐期权合约代码
需要设置环境变量 RQDATAC_LICENSE
"""
import os
import rqdatac as rq

license_key = os.environ.get('RQDATAC_LICENSE')
if not license_key:
    raise ValueError("请设置环境变量 RQDATAC_LICENSE")
rq.init('license', license_key)
print("✓ 米筐连接成功")

# 测试1: 检查合约是否存在
contract_id = 'IO2412C3850'
print(f"\n测试1: 检查合约 {contract_id} 信息")
try:
    instrument = rq.instruments(contract_id)
    print(f"  ✓ 合约存在")
    print(f"  - 标的: {instrument.underlying_symbol}")
    print(f"  - 行权价: {instrument.strike_price}")
    print(f"  - 到期日: {instrument.maturity_date}")
    print(f"  - 期权类型: {instrument.option_type}")
    print(f"  - 行权方式: {instrument.exercise_type}")
except Exception as e:
    print(f"  ✗ 合约不存在或错误: {e}")

# 测试2: 尝试获取价格数据
print(f"\n测试2: 获取价格数据")
try:
    df = rq.get_price(
        order_book_ids=contract_id,
        start_date='20241201',
        end_date='20241220',
        frequency='1d',
        expect_df=True
    )
    print(f"  ✓ 获取成功，共 {len(df)} 条记录")
    print(f"  数据列: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"\n  数据示例:")
        print(df.head())
except Exception as e:
    print(f"  ✗ 获取失败: {e}")

# 测试3: 查询2024年12月的所有期权合约
print(f"\n测试3: 查询2024年12月的沪深300期权合约")
try:
    contracts = rq.options.get_contracts('000300.XSHG', '20241201', '20241220')
    print(f"  ✓ 找到 {len(contracts)} 个合约")
    # 筛选IO2412开头的
    io2412_contracts = [c for c in contracts if c.startswith('IO2412')]
    print(f"  其中 IO2412 合约: {len(io2412_contracts)} 个")
    if len(io2412_contracts) > 0:
        print(f"  示例合约: {io2412_contracts[:5]}")
except Exception as e:
    print(f"  ✗ 查询失败: {e}")

# 测试4: 尝试使用2024年1月的数据（根据notebook里的例子）
print(f"\n测试4: 尝试获取2024年1月数据")
try:
    df = rq.get_price(
        order_book_ids=contract_id,
        frequency='1d',
        expect_df=True
    )
    print(f"  ✓ 获取成功，共 {len(df)} 条记录")
except Exception as e:
    print(f"  ✗ 获取失败: {e}")
