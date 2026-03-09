"""
期权定价模型：Black-Scholes 与 Heston

- BS：使用 hv_20d 作为波动率
- Heston：支持校准与可扩展参数接口（预留神经网络校准通道）

Author: Version_9
"""

from dataclasses import dataclass
from typing import Optional, Callable, Union
import numpy as np
import pandas as pd
from scipy.stats import norm

# 复现用随机种子
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """全局设置随机种子，便于复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


# Heston 依赖 QuantLib
try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False
    ql = None


@dataclass
class HestonParams:
    """Heston 模型参数，为将来神经网络校准预留扩展性"""
    kappa: float   # 波动率均值回归速度
    theta: float   # 长期波动率方差
    sigma: float   # 波动率的波动 (vol-of-vol)
    rho: float     # 标的与波动率的相关系数
    v0: float      # 初始波动率方差

    def to_tuple(self):
        return (self.theta, self.kappa, self.sigma, self.rho, self.v0)


class OptionPricingModel:
    """
    统一期权定价接口：BS + Heston
    """

    def __init__(self, dividend_rate: float = 0.0):
        self.dividend_rate = dividend_rate
        self._heston_params: Optional[HestonParams] = None
        self._heston_params_provider: Optional[Callable[..., HestonParams]] = None

    def set_heston_params_provider(self, provider: Callable[..., HestonParams]) -> None:
        """将来可由神经网络提供参数，支持 (S, T, r, ...) -> HestonParams"""
        self._heston_params_provider = provider

    def clear_heston_params_provider(self) -> None:
        """清除参数提供者，恢复使用校准得到的固定参数"""
        self._heston_params_provider = None

    # -------------------------------------------------------------------------
    # Black-Scholes 定价（使用 hv_20d）
    # -------------------------------------------------------------------------

    def price_bs(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """
        BS 欧式期权定价，sigma 通常来自 hv_20d。

        Parameters:
        -----------
        S : 标的价格
        K : 行权价
        T : 到期时间（年）
        r : 无风险利率
        sigma : 波动率（如 hv_20d）
        option_type : "call" 或 "put"
        """
        if T <= 0:
            # 到期日：直接计算内在价值
            if option_type.lower() == "call":
                return max(0.0, S - K)
            return max(0.0, K - S)

        if sigma <= 0:
            sigma = 1e-8  # 避免除零

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - self.dividend_rate + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        if option_type.lower() == "call":
            price = S * np.exp(-self.dividend_rate * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-self.dividend_rate * T) * norm.cdf(-d1)

        return float(max(0.0, price))

    def price_bs_batch(
        self,
        df: pd.DataFrame,
        S_col: str = "underlying_close",
        K_col: str = "strike_price",
        T_col: str = "time_to_expire",
        r_col: str = "risk_free_rate",
        sigma_col: str = "hv_20d",
        option_col: str = "call_put",
    ) -> np.ndarray:
        """对 DataFrame 批量 BS 定价，call_put: 0=Call, 1=Put"""
        prices = []
        for _, row in df.iterrows():
            opt = "call" if row[option_col] == 0 else "put"
            p = self.price_bs(
                float(row[S_col]),
                float(row[K_col]),
                float(row[T_col]),
                float(row[r_col]),
                float(row[sigma_col]),
                option_type=opt,
            )
            prices.append(p)
        return np.array(prices)

    def implied_volatility_bs(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        sigma_min: float = 1e-6,
        sigma_max: float = 5.0,
    ) -> float:
        """
        从期权价格反算 BS 隐含波动率。使用 scipy.optimize.brentq 求根。

        Parameters:
        -----------
        price : 市场价或预测价
        S, K, T, r : BS 参数
        option_type : "call" 或 "put"
        sigma_min, sigma_max : 搜索边界
        """
        from scipy.optimize import brentq

        def obj(sig):
            return self.price_bs(S, K, T, r, sig, option_type) - price

        if T <= 0 or price <= 0:
            return np.nan
        try:
            iv = brentq(obj, sigma_min, sigma_max)
            return float(iv)
        except (ValueError, RuntimeError):
            return np.nan

    def implied_volatility_bs_batch(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        S_col: str = "underlying_close",
        K_col: str = "strike_price",
        T_col: str = "time_to_expire",
        r_col: str = "risk_free_rate",
        option_col: str = "call_put",
    ) -> np.ndarray:
        """对 DataFrame 批量反算隐含波动率"""
        ivs = []
        for _, row in df.iterrows():
            opt = "call" if row[option_col] == 0 else "put"
            iv = self.implied_volatility_bs(
                float(row[price_col]),
                float(row[S_col]),
                float(row[K_col]),
                float(row[T_col]),
                float(row[r_col]),
                option_type=opt,
            )
            ivs.append(iv)
        return np.array(ivs)

    # -------------------------------------------------------------------------
    # Heston 定价
    # -------------------------------------------------------------------------

    def _get_heston_params(self, S: float, T: float, r: float, **kwargs) -> HestonParams:
        """获取 Heston 参数：优先使用 params_provider，否则使用校准参数"""
        if self._heston_params_provider is not None:
            return self._heston_params_provider(S=S, T=T, r=r, **kwargs)
        if self._heston_params is None:
            raise ValueError("Heston 参数未校准，请先调用 calibrate_heston 或 set_heston_params")
        return self._heston_params

    def set_heston_params(self, params: HestonParams) -> None:
        """直接设置 Heston 参数（校准后调用）"""
        self._heston_params = params

    def price_heston(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        params: Optional[HestonParams] = None,
        option_type: str = "call",
    ) -> float:
        """
        Heston 欧式期权定价。

        Parameters:
        -----------
        params : 若为 None，使用校准参数或 params_provider
        """
        if not HAS_QUANTLIB:
            raise ImportError("Heston 定价需要 QuantLib，请安装: pip install QuantLib-Python")

        p = params if params is not None else self._get_heston_params(S, T, r)

        if T <= 0:
            if option_type.lower() == "call":
                return max(0.0, S - K)
            return max(0.0, K - S)

        day_count = ql.Actual365Fixed()
        calendar = ql.China()
        calc_date = ql.Date(1, 1, 2024)  # 占位，仅用于构建
        ql.Settings.instance().evaluationDate = calc_date

        yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, r, day_count))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, self.dividend_rate, day_count))
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))

        process = ql.HestonProcess(
            yield_ts, dividend_ts, spot_handle,
            p.v0, p.kappa, p.theta, p.sigma, p.rho
        )
        model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model)
        days_to_expiry = max(1, int(round(T * 365)))
        expiry_date = calc_date + ql.Period(days_to_expiry, ql.Days)
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call if option_type.lower() == "call" else ql.Option.Put, K),
            ql.EuropeanExercise(expiry_date)
        )
        option.setPricingEngine(engine)
        price = option.NPV()
        return float(max(0.0, price))

    def price_heston_batch(
        self,
        df: pd.DataFrame,
        S_col: str = "underlying_close",
        K_col: str = "strike_price",
        T_col: str = "time_to_expire",
        r_col: str = "risk_free_rate",
        option_col: str = "call_put",
    ) -> np.ndarray:
        """对 DataFrame 批量 Heston 定价"""
        prices = []
        for _, row in df.iterrows():
            opt = "call" if row[option_col] == 0 else "put"
            p = self.price_heston(
                float(row[S_col]),
                float(row[K_col]),
                float(row[T_col]),
                float(row[r_col]),
                option_type=opt,
            )
            prices.append(p)
        return np.array(prices)

    # -------------------------------------------------------------------------
    # Heston 校准
    # -------------------------------------------------------------------------

    def calibrate_heston(
        self,
        market_data: pd.DataFrame,
        S_col: str = "underlying_close",
        K_col: str = "strike_price",
        T_col: str = "time_to_expire",
        r_col: str = "risk_free_rate",
        price_col: str = "close",
        option_col: str = "call_put",
        init_params: Optional[HestonParams] = None,
        method: str = "least_squares",
        max_samples: int = 2000,
    ) -> HestonParams:
        """
        使用市场价校准 Heston 参数。

        Parameters:
        -----------
        market_data : 包含 S, K, T, r, 市场价, option_type 的 DataFrame
        method : "least_squares" 或 "differential_evolution"
        max_samples : 校准时的最大样本数（过多会慢）
        """
        if not HAS_QUANTLIB:
            raise ImportError("Heston 校准需要 QuantLib，请安装: pip install QuantLib-Python")

        df = market_data.dropna(subset=[S_col, K_col, T_col, r_col, price_col])
        df = df[df[T_col] > 1e-6]  # 过滤 T<=0
        df = df[df[price_col] > 1e-6]  # 过滤无效价格

        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=SEED)

        S_ref = float(df[S_col].iloc[0])
        r_ref = float(df[r_col].iloc[0])

        init = init_params or HestonParams(
            kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04
        )
        x0 = np.array(list(init.to_tuple()))
        bounds = [(0.01, 1), (0.01, 15), (0.01, 1), (-0.99, 0.99), (0.01, 1)]

        def objective(x: np.ndarray) -> np.ndarray:
            theta, kappa, sigma, rho, v0 = x
            params = HestonParams(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0)
            self._heston_params = params
            errors = []
            for _, row in df.iterrows():
                try:
                    pred = self.price_heston(
                        float(row[S_col]), float(row[K_col]), float(row[T_col]),
                        float(row[r_col]), params=params,
                        option_type="call" if row[option_col] == 0 else "put",
                    )
                    market_p = float(row[price_col])
                    errors.append((pred - market_p) / (market_p + 1e-8))
                except Exception:
                    errors.append(1e6)
            return np.array(errors)

        if method == "least_squares":
            from scipy.optimize import least_squares
            res = least_squares(objective, x0, bounds=([b[0] for b in bounds], [b[1] for b in bounds]))
            x_opt = res.x
        else:
            from scipy.optimize import differential_evolution
            def scalar_obj(x):
                return np.sqrt(np.mean(objective(x) ** 2))
            res = differential_evolution(scalar_obj, bounds, maxiter=80, seed=SEED, polish=True)
            x_opt = res.x

        theta, kappa, sigma, rho, v0 = x_opt
        self._heston_params = HestonParams(
            kappa=float(kappa), theta=float(theta), sigma=float(sigma),
            rho=float(rho), v0=float(v0)
        )
        return self._heston_params


def extract_expiry_yymm(order_book_id: str) -> str:
    """从 order_book_id 提取 YYMM，如 IO2401C2900 -> 2401"""
    import re
    m = re.search(r"IO(\d{4})[CP]", str(order_book_id))
    return m.group(1) if m else ""


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算 MAE, RMSE, MAPE"""
    mask = y_true > 1e-8
    if mask.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    yt, yp = y_true[mask], y_pred[mask]
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def compute_daily_atm_iv(df: pd.DataFrame) -> pd.Series:
    """
    计算每个交易日的ATM隐含波动率（用于v0初始化）
    
    策略：找到每日最接近平值的期权（|ln(S/K)|最小），取其iv中位数
    
    Parameters:
    -----------
    df : 必须包含 ['date', 'underlying_close', 'strike_price', 'iv'] 列
    
    Returns:
    --------
    pd.Series : index=date, value=atm_iv（年化标准差）
    
    Notes:
    ------
    - 每日选取最接近ATM的5个期权（log moneyness最小）
    - 使用中位数避免异常值影响
    - 直接使用数据中的iv列，无需重新计算BS隐含波动率
    """
    df_copy = df.copy()
    
    # 计算绝对log moneyness（衡量期权距离平值的程度）
    df_copy['abs_log_moneyness'] = np.abs(
        np.log(df_copy['underlying_close'] / df_copy['strike_price'])
    )
    
    daily_atm_iv = []
    for date, group in df_copy.groupby('date'):
        # 找到最接近ATM的期权（log moneyness接近0）
        atm_options = group.nsmallest(5, 'abs_log_moneyness')
        atm_iv = atm_options['iv'].median()  # 使用中位数避免异常值
        daily_atm_iv.append({'date': date, 'atm_iv': atm_iv})
    
    return pd.DataFrame(daily_atm_iv).set_index('date')['atm_iv']
