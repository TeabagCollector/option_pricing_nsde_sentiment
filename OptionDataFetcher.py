"""
期权数据获取模块
通过米筐API获取期权完整数据和Greeks信息
"""

"""
Option data fetcher module.
Requires RQDATAC_LICENSE env var (RiceQuant/Miqiang API).
"""

import os
import rqdatac as rq
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
import warnings
warnings.filterwarnings('ignore')

class OptionDataFetcher:
    """
    期权数据获取类 - 专注于期权定价所需的核心数据
    
    功能:
    1. 获取期权价格数据（open, close, high, low, volume等）
    2. 获取期权Greeks（delta, gamma, vega, theta, rho, iv）
    3. 获取标的资产价格和历史波动率
    4. 获取无风险利率
    5. 自动保存/加载数据到本地
    
    数据字段说明:
    - K (strike_price): 行权价
    - S (underlying_close): 标的资产价格
    - T (time_to_expire): 到期时间（年化）
    - r (risk_free_rate): 无风险利率
    - sigma (hv_20d / iv): 波动率（历史波动率 / 隐含波动率）
    - call_put: 期权类型（0=看涨, 1=看跌）
    - euro_american: 行权方式（0=欧式, 1=美式）
    
    使用示例:
    >>> fetcher = OptionDataFetcher(license='your_license')
    >>> fetcher.init_connection()
    >>> df = fetcher.get_option_data('IO2412C3850', start_date='20240101', end_date='20241231')
    >>> fetcher.save_data(df, 'option_data.csv')
    """
    
    def __init__(self, license: Optional[str] = None, data_dir: Optional[str] = None):
        """
        初始化数据获取器
        
        Args:
            license: 米筐API的license key，如果不提供则需要后续调用init_connection时提供
            data_dir: 数据保存目录，默认为当前目录下的data文件夹
        """
        self.license = license
        self.is_connected = False
        
        # 设置数据保存目录（使用绝对路径）
        if data_dir:
            self.data_dir = Path(data_dir).resolve()
        else:
            self.data_dir = Path(__file__).parent.resolve() / 'data'
        
        # 如果目录不存在则创建
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def init_connection(self, license: Optional[str] = None):
        """
        初始化米筐数据连接
        
        Args:
            license: 米筐API的license key。若不提供，则从环境变量 RQDATAC_LICENSE 读取
        """
        if license:
            self.license = license
        elif not self.license:
            self.license = os.environ.get('RQDATAC_LICENSE')
            
        if not self.license:
            raise ValueError("请提供米筐API的license key（参数或环境变量 RQDATAC_LICENSE）")
            
        try:
            rq.init('license', self.license)
            self.is_connected = True
            print("✓ 米筐数据连接成功")
        except Exception as e:
            print(f"✗ 米筐数据连接失败: {e}")
            raise
            
    def _check_connection(self):
        """检查是否已连接"""
        if not self.is_connected:
            raise ConnectionError("请先调用 init_connection() 初始化数据连接")
    
    def get_option_data(self, 
                       order_book_id: Union[str, List[str]], 
                       start_date: str = '20230101', 
                       end_date: str = '20241231',
                       frequency: str = '1d',
                       include_greeks: bool = True,
                       greeks_fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取期权完整数据
        
        Args:
            order_book_id: 期权合约代码，支持单个或多个
            start_date: 开始日期，格式'YYYYMMDD'
            end_date: 结束日期，格式'YYYYMMDD'
            frequency: 数据频率，默认'1d'（日频）
            include_greeks: 是否包含Greeks数据
            greeks_fields: 需要获取的Greeks字段列表，默认获取所有
                          可选: ['delta', 'gamma', 'vega', 'theta', 'rho', 'iv']
        
        Returns:
            df: 包含完整期权信息和Greeks的DataFrame
        """
        self._check_connection()
        
        # 如果是单个合约，转为列表
        if isinstance(order_book_id, str):
            order_book_ids = [order_book_id]
        else:
            order_book_ids = order_book_id
            
        print(f"开始获取 {len(order_book_ids)} 个合约的数据...")
        
        all_data = []
        
        for idx, contract_id in enumerate(order_book_ids):
            print(f"  [{idx+1}/{len(order_book_ids)}] 正在获取 {contract_id} ...")
            
            try:
                # 1. 获取期权基础价格数据（保留所有字段）
                df_price = rq.get_price(
                    order_book_ids=contract_id,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    adjust_type='none',
                    expect_df=True
                )
                
                if df_price.empty:
                    print(f"    ⚠ {contract_id} 无数据，跳过")
                    continue
                
                # 2. 获取合约详细信息
                instrument = rq.instruments(contract_id)
                
                # 添加期权定价必需的静态信息
                df_price['strike_price'] = instrument.strike_price  # 行权价 K
                df_price['maturity_date'] = instrument.maturity_date  # 到期日
                df_price['underlying_symbol'] = instrument.underlying_symbol  # 标的代码
                df_price['contract_multiplier'] = instrument.contract_multiplier  # 合约乘数
                
                # 从合约代码判断期权类型 (C=看涨, P=看跌)
                df_price['option_type'] = 'Call' if 'C' in contract_id else 'Put'
                
                # 从API获取行权方式 ('E'=欧式, 'A'=美式)
                df_price['exercise_type'] = instrument.exercise_type
                
                # 3. 计算到期时间（年化）
                df_price['time_to_expire'] = [
                    instrument.days_to_expire(date) / 365 
                    for date in df_price.index.get_level_values('date')
                ]
                
                # 4. 获取Greeks数据
                if include_greeks:
                    if greeks_fields is None:
                        # 默认获取所有Greeks字段
                        greeks_fields = ['delta', 'gamma', 'vega', 'theta', 'rho', 'iv']
                    
                    df_greeks = rq.options.get_greeks(
                        order_book_ids=contract_id,
                        start_date=start_date,
                        end_date=end_date,
                        model='last',  # 使用最新价计算
                        fields=greeks_fields,
                        frequency=frequency
                    )
                    
                    if not df_greeks.empty:
                        # 合并Greeks数据
                        # 关键修复：重命名trading_date索引为date，确保索引对齐
                        df_greeks.index = df_greeks.index.rename(['order_book_id', 'date'])
                        df_price = df_price.join(df_greeks, how='left')
                
                # 5. 获取标的资产价格
                # 注意: IO期权的underlying_symbol是'IO'，需要转换为实际标的'000300.XSHG'
                underlying_symbol = instrument.underlying_symbol
                if underlying_symbol == 'IO':
                    actual_underlying = '000300.XSHG'  # 沪深300指数
                elif underlying_symbol == 'HO':
                    actual_underlying = '510050.XSHG'  # 50ETF
                elif underlying_symbol == 'MO':
                    actual_underlying = '510300.XSHG'  # 300ETF
                else:
                    actual_underlying = underlying_symbol
                
                df_underlying = rq.get_price(
                    order_book_ids=actual_underlying,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    fields=['close', 'open', 'high', 'low'],
                    adjust_type='none',
                    expect_df=True
                )
                
                if not df_underlying.empty:
                    df_underlying = df_underlying.droplevel('order_book_id')
                    df_underlying = df_underlying.rename(columns={
                        'close': 'underlying_close',
                        'open': 'underlying_open',
                        'high': 'underlying_high',
                        'low': 'underlying_low'
                    })
                    df_price = df_price.join(df_underlying, how='left')
                
                # 6. 计算历史波动率（20日）
                try:
                    # 获取更长时间段的数据用于计算波动率（至少需要20+天）
                    # 向前推30个交易日以确保有足够数据
                    from datetime import datetime, timedelta
                    if isinstance(start_date, str):
                        start_dt = pd.to_datetime(start_date)
                    else:
                        start_dt = pd.to_datetime(str(start_date))
                    
                    # 向前推60天以确保有足够的交易日
                    extended_start = (start_dt - timedelta(days=60)).strftime('%Y%m%d')
                    
                    df_hv_raw = rq.get_price(
                        order_book_ids=actual_underlying,
                        start_date=extended_start,
                        end_date=end_date,
                        frequency='1d',  # 波动率用日频计算
                        fields=['prev_close', 'close'],
                        adjust_type='none',
                        expect_df=True
                    )
                    
                    if not df_hv_raw.empty:
                        df_hv_raw = df_hv_raw.droplevel('order_book_id')
                        # 计算对数收益率
                        df_hv_raw['log_return'] = np.log(df_hv_raw['close'] / df_hv_raw['prev_close'])
                        # 计算20日历史波动率（年化）
                        window = 20
                        df_hv_raw['hv_20d'] = df_hv_raw['log_return'].rolling(window=window).std() * np.sqrt(252)
                        
                        # 只保留hv_20d列
                        df_hv = df_hv_raw[['hv_20d']]
                        df_price = df_price.join(df_hv, how='left')
                except Exception as e:
                    print(f"    ⚠ 无法计算 {actual_underlying} 的历史波动率: {e}")
                
                # 7. 获取无风险利率（10年期国债收益率）
                try:
                    df_rate = rq.get_yield_curve(
                        start_date=start_date,
                        end_date=end_date,
                        tenor='10Y'
                    )
                    
                    if not df_rate.empty:
                        df_rate = df_rate.rename(columns={'10Y': 'risk_free_rate'})
                        df_rate.index.name = 'date'
                        # 已经是小数形式，如0.025表示2.5%，无需转换
                        df_price = df_price.join(df_rate, how='left')
                except Exception as e:
                    print(f"    ⚠ 无法获取无风险利率: {e}")
                
                # 8. 添加衍生特征（用于模型输入）
                df_price['moneyness(S/K)'] = df_price['underlying_close'] / df_price['strike_price']  # 实值程度
                df_price['log_moneyness(ln(S/K))'] = np.log(df_price['moneyness(S/K)'])  # 对数实值程度
                df_price['moneyness(K/S)'] = df_price['strike_price'] / df_price['underlying_close']  # 实值程度
                df_price['log_moneyness(ln(K/S))'] = np.log(df_price['moneyness(K/S)'])  # 对数实值程度
                df_price['call_put'] = df_price['option_type'].map({'Call': 0, 'Put': 1})  # 数值化: 看涨=0, 看跌=1
                df_price['euro_american'] = df_price['exercise_type'].map({'E': 0, 'A': 1})  # 数值化: 欧式=0, 美式=1

                all_data.append(df_price)
                print(f"    ✓ {contract_id} 数据获取成功，共 {len(df_price)} 条记录")
                
            except Exception as e:
                print(f"    ✗ {contract_id} 数据获取失败: {e}")
                continue
        
        if not all_data:
            raise ValueError("所有合约都没有获取到数据")
        
        # 合并所有数据
        df_final = pd.concat(all_data)
        print(f"\n✓ 数据获取完成！总共 {len(df_final)} 条记录，涵盖 {len(order_book_ids)} 个合约")
        
        return df_final
    
    def get_contracts_by_underlying(self, 
                                   underlying: str = '000300.XSHG',
                                   start_date: str = '20240101',
                                   end_date: str = '20241231') -> List[str]:
        """
        根据标的代码获取期权合约列表
        
        Args:
            underlying: 标的资产代码，如'000300.XSHG'（沪深300）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            合约代码列表
        """
        self._check_connection()
        
        contracts = rq.options.get_contracts(underlying, start_date, end_date)
        print(f"✓ 找到 {len(contracts)} 个期权合约")
        return contracts
    
    def get_monthly_contracts(self, 
                             year: int = 2024,
                             underlying: str = '000300.XSHG',
                             option_type: Optional[str] = None) -> dict:
        """
        获取指定年份每月到期的期权合约列表
        
        Args:
            year: 年份，如2024
            underlying: 标的资产代码，如'000300.XSHG'（沪深300）
            option_type: 期权类型，'C'=看涨, 'P'=看跌, None=全部
            
        Returns:
            字典，格式为 {'2024-01': ['IO2401C3800', ...], '2024-02': [...], ...}
        """
        self._check_connection()
        
        monthly_contracts = {}
        
        for month in range(1, 13):
            month_str = f"{year}-{month:02d}"
            maturity = f"{str(year)[2:]}{month:02d}"  # 如 '2401' 代表2024年1月
            
            print(f"  查询 {month_str} 到期的合约...")
            
            try:
                # 使用米筐API查询指定到期月份的合约
                contracts = rq.options.get_contracts(
                    underlying=underlying,
                    maturity=maturity,
                    option_type=option_type
                )
                
                if contracts:
                    monthly_contracts[month_str] = contracts
                    print(f"    ✓ 找到 {len(contracts)} 个合约")
                else:
                    print(f"    - 无合约")
                    
            except Exception as e:
                print(f"    ✗ 查询失败: {e}")
                continue
        
        print(f"\n✓ 共找到 {len(monthly_contracts)} 个月份的合约，总计 {sum(len(v) for v in monthly_contracts.values())} 个合约")
        return monthly_contracts
    
    def get_contract_trading_dates(self, order_book_id: str) -> tuple:
        """
        获取合约的实际交易日期范围
        
        Args:
            order_book_id: 期权合约代码
            
        Returns:
            (上市日期, 到期日期) 的元组，格式为 'YYYYMMDD'
        """
        self._check_connection()
        
        try:
            instrument = rq.instruments(order_book_id)
            
            # 获取上市日期
            listed_date = instrument.listed_date
            if isinstance(listed_date, str):
                listed_date = pd.to_datetime(listed_date).strftime('%Y%m%d')
            else:
                listed_date = listed_date.strftime('%Y%m%d')
            
            # 获取到期日期
            maturity_date = instrument.maturity_date
            if isinstance(maturity_date, str):
                maturity_date = pd.to_datetime(maturity_date).strftime('%Y%m%d')
            else:
                maturity_date = maturity_date.strftime('%Y%m%d')
            
            return (listed_date, maturity_date)
            
        except Exception as e:
            print(f"⚠ 无法获取 {order_book_id} 的交易日期: {e}")
            return (None, None)
    
    def get_annual_data(self,
                       year: int = 2024,
                       underlying: str = '000300.XSHG',
                       option_type: Optional[str] = None,
                       save_monthly: bool = True,
                       include_greeks: bool = True,
                       frequency: str = '1d') -> pd.DataFrame:
        """
        获取指定年份的所有期权数据（按月到期）
        
        Args:
            year: 年份，如2024
            underlying: 标的资产代码，如'000300.XSHG'（沪深300）
            option_type: 期权类型，'C'=看涨, 'P'=看跌, None=全部
            save_monthly: 是否按月保存数据（推荐，避免单文件过大）
            include_greeks: 是否包含Greeks数据
            frequency: 数据频率，默认'1d'
            
        Returns:
            包含全年数据的DataFrame
        """
        self._check_connection()
        
        print(f"="*60)
        print(f"开始获取 {year} 年期权数据")
        print(f"="*60)
        
        # 1. 获取每月到期的合约列表
        print(f"\n[步骤 1/3] 查询各月到期的合约列表...")
        monthly_contracts = self.get_monthly_contracts(
            year=year,
            underlying=underlying,
            option_type=option_type
        )
        
        if not monthly_contracts:
            raise ValueError(f"{year}年没有找到任何期权合约")
        
        # 2. 逐月获取数据
        print(f"\n[步骤 2/3] 逐月获取期权数据...")
        all_monthly_data = []
        
        for month_idx, (month_str, contracts) in enumerate(monthly_contracts.items(), 1):
            print(f"\n--- {month_str} ({month_idx}/{len(monthly_contracts)}) ---")
            print(f"  合约数量: {len(contracts)}")
            
            # 获取该月份合约的交易日期范围
            # 使用第一个合约作为参考，获取大致的日期范围
            first_contract = contracts[0]
            listed_date, maturity_date = self.get_contract_trading_dates(first_contract)
            
            if not listed_date or not maturity_date:
                print(f"  ⚠ 无法获取交易日期，跳过该月")
                continue
            
            print(f"  交易日期范围: {listed_date} ~ {maturity_date}")
            
            try:
                # 获取该月所有合约的数据
                df_month = self.get_option_data(
                    order_book_id=contracts,
                    start_date=listed_date,
                    end_date=maturity_date,
                    frequency=frequency,
                    include_greeks=include_greeks
                )
                
                all_monthly_data.append(df_month)
                
                # 如果设置了按月保存，则保存当前月份的数据
                if save_monthly:
                    filename = f"annual_{year}_{month_str.replace('-', '')}_data.csv"
                    self.save_data(df_month, filename, subfolder='annual')
                    print(f"  ✓ 已保存到: annual/{filename}")
                
            except Exception as e:
                print(f"  ✗ {month_str} 数据获取失败: {e}")
                continue
        
        # 3. 合并所有月份的数据
        print(f"\n[步骤 3/3] 合并所有数据...")
        
        if not all_monthly_data:
            raise ValueError("所有月份的数据都获取失败")
        
        df_annual = pd.concat(all_monthly_data)
        
        # 保存完整的年度数据
        filename_annual = f"annual_{year}_complete.csv"
        self.save_data(df_annual, filename_annual, subfolder='annual')
        
        print(f"\n" + "="*60)
        print(f"✓ {year} 年数据获取完成！")
        print(f"  总记录数: {len(df_annual)}")
        print(f"  合约数量: {df_annual.index.get_level_values('order_book_id').nunique()}")
        print(f"  日期范围: {df_annual.index.get_level_values('date').min()} ~ {df_annual.index.get_level_values('date').max()}")
        print(f"  已保存到: annual/{filename_annual}")
        print(f"="*60)
        
        return df_annual
    
    def save_data(self, df: pd.DataFrame, filename: str, format: str = 'csv', subfolder: Optional[str] = None):
        """
        保存数据到本地
        
        Args:
            df: 要保存的DataFrame
            filename: 文件名
            format: 保存格式，支持'csv', 'parquet', 'pickle'
            subfolder: 子文件夹名称，如'raw', 'processed'等，用于分类管理
        """
        # 如果指定了子文件夹，则在data目录下创建
        if subfolder:
            save_dir = self.data_dir / subfolder
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.data_dir
        
        filepath = save_dir / filename
        
        if format == 'csv':
            df.to_csv(filepath)
        elif format == 'parquet':
            df.to_parquet(filepath)
        elif format == 'pickle':
            df.to_pickle(filepath)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        print(f"✓ 数据已保存到: {filepath}")
        
    def load_data(self, filename: str, format: str = 'csv', subfolder: Optional[str] = None) -> pd.DataFrame:
        """
        从本地加载数据
        
        Args:
            filename: 文件名
            format: 文件格式
            subfolder: 子文件夹名称
            
        Returns:
            DataFrame
        """
        if subfolder:
            filepath = self.data_dir / subfolder / filename
        else:
            filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        if format == 'csv':
            df = pd.read_csv(filepath, index_col=[0, 1], parse_dates=[0])
        elif format == 'parquet':
            df = pd.read_parquet(filepath)
        elif format == 'pickle':
            df = pd.read_pickle(filepath)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        print(f"✓ 数据已加载: {filepath}")
        return df


# 使用示例
if __name__ == '__main__':
    # 示例1: 获取单个合约数据
    fetcher = OptionDataFetcher(data_dir="Version_9/outputs/TEST")
    
    fetcher.init_connection()
    
    # 获取指定合约数据（注意：IO2412合约在2024年12月才上市）
    df = fetcher.get_option_data(
        order_book_id='IO2412C3850',
        start_date='20231201',  # 用更早的日期检查信息获取全面性+是否报错
        end_date='20241220',
        include_greeks=True
    )
    
    # 保存数据到raw子文件夹（原始数据）
    fetcher.save_data(df, 'IO2412C3850_20241201_20241220.csv', subfolder='raw')
    print(f"\n✓ 单个合约数据获取完成")
    print(f"  数据字段: {df.columns.tolist()}")
    print(f"\n  数据预览:")
    print(df.head())
    
    # 示例2: 批量获取多个合约
    print("\n" + "="*60)
    print("示例2: 批量获取多个合约")
    contracts = ['IO2412C3850', 'IO2412P3850', 'IO2412C4000']
    df_multiple = fetcher.get_option_data(
        order_book_id=contracts,
        start_date='20241201',
        end_date='20241220'
    )
    
    # 保存到raw子文件夹，使用描述性文件名
    fetcher.save_data(df_multiple, 'IO2412_multiple_20241201_20241220.csv', subfolder='raw')
    print(f"\n✓ 批量合约数据获取完成，共 {len(df_multiple)} 条记录")
    
    print("\n" + "="*60)
    print("数据已按以下结构保存:")
    print("  data/")
    print("    ├── raw/                    # 原始数据")
    print("    │   ├── IO2412C3850_20241201_20241220.csv")
    print("    │   └── IO2412_multiple_20241201_20241220.csv")
    print("    ├── processed/              # 可以存放处理后的数据")
    print("    └── models/                 # 可以存放训练好的模型")
    print("="*60)
