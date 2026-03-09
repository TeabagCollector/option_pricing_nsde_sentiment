"""
神经随机微分方程（Neural SDE）期权定价器

参考 Wang & Hong (2021)，使用神经网络参数化 Heston 模型的漂移和扩散项，
并整合情绪因子 z，实现动力学层面的情绪建模。

标的资产价格过程：
dS_t = μ_S(S_t, v_t, r, t, z) dt + σ_S(S_t, v_t, r, t, z) dW_t^S

波动率过程：
dv_t = μ_v(S_t, v_t, r, t, z) dt + σ_v(S_t, v_t, r, t, z) dW_t^v

其中 μ_S, σ_S, μ_v, σ_v 均由神经网络参数化。

Author: Version_9
Date: 2026-02-17
"""

from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

from OptionPricingModel import set_seed, SEED, compute_metrics, HestonParams
from FlexibleMLP import FlexibleMLP


class NeuralSDEPricer:
    """
    神经随机微分方程期权定价器
    
    使用 4 个神经网络建模 Heston 过程的漂移和扩散项：
    - net_mu_S: 标的资产漂移 μ_S(S, v, r, t, z)
    - net_sigma_S: 标的资产扩散 σ_S(S, v, r, t, z)
    - net_mu_v: 波动率漂移 μ_v(S, v, r, t, z)
    - net_sigma_v: 波动率扩散 σ_v(S, v, r, t, z)
    """
    
    def __init__(
        self,
        latent_dim: int = 2,        # 情绪因子维度
        hidden_dims: list = [32, 32],
        n_paths: int = 5000,        # 蒙特卡洛路径数（训练时）
        n_steps: int = 50,          # 时间离散步数
        rho: Optional[float] = -0.5,  # S 和 v 的相关系数（None时使用rho_source）
        rho_source: str = 'fixed',  # rho来源：'fixed', 'heston'
        heston_params: Optional[HestonParams] = None,  # 基准 Heston 参数
        residual_scale: float = 0.3,  # 残差缩放因子
        loss_type: str = 'mse',     # 损失函数类型：'mse', 'mape', 'relative_mse'
        v0_source: str = 'hv_20d',  # 初始方差来源：'hv_20d', 'atm_iv', 'iv', 'heston'
        random_state: Optional[int] = None,
    ):
        """
        Parameters:
        -----------
        latent_dim : 情绪因子维度（来自 VAE）
        hidden_dims : 隐藏层维度列表
        n_paths : 蒙特卡洛路径数
        n_steps : 时间离散步数
        rho : 标的价格和波动率的相关系数（当rho_source='fixed'时使用）
        rho_source : rho数据源
            - 'fixed': 使用rho参数指定的固定值
            - 'heston': 使用Heston校准得到的rho值
        heston_params : 基准 Heston 参数（可选，训练时自动校准）
        residual_scale : 残差缩放因子（tanh 输出乘以此系数）
        loss_type : 损失函数类型，支持 'mse', 'mape', 'relative_mse'
        v0_source : 初始方差数据源
            - 'hv_20d': 历史波动率的平方
            - 'atm_iv': 每日ATM隐含波动率的平方
            - 'iv': 单个期权的隐含波动率的平方
            - 'heston': 使用Heston校准得到的v0值
        random_state : 随机种子
        """
        if not HAS_TORCH:
            raise ImportError("NeuralSDEPricer 需要 PyTorch")
        
        self.latent_dim = latent_dim
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.rho = rho
        self.rho_source = rho_source
        self.heston_params = heston_params  # 基准 Heston 参数
        self.residual_scale = residual_scale  # 残差缩放因子
        self.loss_type = loss_type  # 损失函数类型
        self.v0_source = v0_source  # 初始方差来源
        self.random_state = random_state or SEED
        
        # 输入维度：(S, v, r, t, z_1, z_2, ...) = 4 + latent_dim
        input_dim = 4 + latent_dim
        
        # 4 个神经网络（全部使用 tanh 激活，输出范围 [-1, 1]）
        # Δμ_S: 标的资产漂移残差
        self.net_mu_S = FlexibleMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='tanh',  # 允许正负残差
            dropout=0.0,
            random_state=self.random_state
        )
        
        # Δσ_S: 标的资产扩散残差
        self.net_sigma_S = FlexibleMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='tanh',  # 允许正负残差
            dropout=0.0,
            random_state=self.random_state + 1
        )
        
        # Δμ_v: 波动率漂移残差
        self.net_mu_v = FlexibleMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='tanh',
            dropout=0.0,
            random_state=self.random_state + 2
        )
        
        # Δσ_v: 波动率扩散残差
        self.net_sigma_v = FlexibleMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='tanh',  # 允许正负残差
            dropout=0.0,
            random_state=self.random_state + 3
        )
        
        self.history_: Optional[dict] = None
    
    @staticmethod
    def _compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
        """
        计算损失函数
        
        Parameters:
        -----------
        pred : 预测值
        target : 真实值
        loss_type : 损失函数类型
            - 'mse': 均方误差（适合高价期权）
            - 'mape': 平均绝对百分比误差（推荐，对所有价格区间公平）
            - 'relative_mse': 相对均方误差（介于mse和mape之间）
        
        Returns:
        --------
        loss : 损失值
        """
        if loss_type == 'mse':
            return torch.mean((pred - target) ** 2)
        elif loss_type == 'mape':
            return torch.mean(torch.abs((pred - target) / (target + 1e-8)))
        elif loss_type == 'relative_mse':
            return torch.mean(((pred - target) / (target + 1e-8)) ** 2)
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}. 支持的类型: 'mse', 'mape', 'relative_mse'")
    
    def _get_rho(self) -> float:
        """
        获取当前使用的rho值
        
        Returns:
        --------
        rho : 相关系数
        """
        if self.rho_source == 'fixed':
            if self.rho is None:
                raise ValueError("rho_source='fixed' 但 rho 参数为 None")
            return self.rho
        elif self.rho_source == 'heston':
            if self.heston_params is None:
                raise ValueError("rho_source='heston' 需要提供 heston_params 或在训练时自动校准")
            return self.heston_params.rho
        else:
            raise ValueError(f"不支持的 rho_source: {self.rho_source}. 支持的选项: 'fixed', 'heston'")
    
    def simulate_paths(
        self,
        S0: float,
        v0: float,
        r: float,
        T: float,
        z: np.ndarray,
        n_paths: Optional[int] = None,
        return_torch: bool = False  # 新增参数：是否返回PyTorch tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Euler-Maruyama 方法模拟资产价格路径
        
        Parameters:
        -----------
        S0 : 初始标的价格
        v0 : 初始波动率方差
        r : 无风险利率
        T : 到期时间
        z : 情绪因子 (latent_dim,)
        n_paths : 路径数（None 则使用 self.n_paths）
        return_torch : 是否返回PyTorch tensor（用于训练）
        
        Returns:
        --------
        S_paths : (n_paths, n_steps+1) 价格路径
        v_paths : (n_paths, n_steps+1) 波动率路径
        """
        if n_paths is None:
            n_paths = self.n_paths
        
        # 获取当前使用的rho值
        rho = self._get_rho()
        
        dt = T / self.n_steps
        
        if return_torch:
            # 训练模式：使用PyTorch保持梯度
            # 只保留当前状态，不存储路径（避免就地操作问题）
            sqrt_dt = torch.sqrt(torch.tensor(dt, dtype=torch.float32))
            
            # 预生成所有随机数（避免循环中生成）
            # 任务 3：引入对偶变量法 (Antithetic Variates) 降低方差
            
            # if self.random_state is not None:  # 在这里设置随机种子好像会导致蒙特卡洛路径完全一致，暂时先不设置随机种子
            #     torch.manual_seed(self.random_state)
            
            half_paths = n_paths // 2
            n_paths = half_paths * 2  # 强制偶数
            Z_S = torch.randn(self.n_steps, half_paths)
            all_dW_S = torch.cat([Z_S, -Z_S], dim=1)
            Z_v = torch.randn(self.n_steps, half_paths)
            all_dW_v_indep = torch.cat([Z_v, -Z_v], dim=1)
            
            # 更新初始状态为偶数路径
            S_t = torch.full((n_paths,), S0, dtype=torch.float32)
            v_t = torch.full((n_paths,), v0, dtype=torch.float32)
            
            # 预计算scaler（避免重复创建tensor）
            scaler_mean_mu_S = torch.tensor(self.net_mu_S._scaler_mean, dtype=torch.float32)
            scaler_std_mu_S = torch.tensor(self.net_mu_S._scaler_std, dtype=torch.float32) + 1e-8
            scaler_mean_sigma_S = torch.tensor(self.net_sigma_S._scaler_mean, dtype=torch.float32)
            scaler_std_sigma_S = torch.tensor(self.net_sigma_S._scaler_std, dtype=torch.float32) + 1e-8
            scaler_mean_mu_v = torch.tensor(self.net_mu_v._scaler_mean, dtype=torch.float32)
            scaler_std_mu_v = torch.tensor(self.net_mu_v._scaler_std, dtype=torch.float32) + 1e-8
            scaler_mean_sigma_v = torch.tensor(self.net_sigma_v._scaler_mean, dtype=torch.float32)
            scaler_std_sigma_v = torch.tensor(self.net_sigma_v._scaler_std, dtype=torch.float32) + 1e-8
            z_tensor = torch.tensor(z, dtype=torch.float32)
            rho_complement = torch.sqrt(torch.tensor(1 - rho**2 + 1e-8))
            
            # Euler-Maruyama 迭代
            for i in range(self.n_steps):
                t = i * dt
                # 使用 max() 替代 clamp 以保留梯度（对于单边界约束）
                v_t_safe = torch.max(v_t, torch.tensor(1e-6, dtype=torch.float32))
                
                # 构造输入特征（任务 2：使用 Moneyness 归一化 S_t）
                features = torch.stack([
                    S_t / S0,  # Moneyness 归一化，避免 tanh 饱和
                    v_t_safe,
                    torch.full((n_paths,), r, dtype=torch.float32),
                    torch.full((n_paths,), t, dtype=torch.float32),
                    z_tensor[0].expand(n_paths).clone(),
                    (z_tensor[1].expand(n_paths).clone() if len(z) > 1 else torch.zeros(n_paths))
                ], dim=1)
                
                # 1. 神经网络预测残差（显式 tanh 包裹输出）
                # 任务 2：强制 delta_mu_S = 0 以满足风险中性定价
                delta_mu_S = torch.zeros(n_paths, dtype=torch.float32)
                
                # 显式 tanh 包裹模型输出
                # delta_sigma_S = (torch.tanh(self.net_sigma_S._model((features - scaler_mean_sigma_S) / scaler_std_sigma_S).squeeze(-1)) * self.residual_scale).clone()
                # delta_mu_v = (torch.tanh(self.net_mu_v._model((features - scaler_mean_mu_v) / scaler_std_mu_v).squeeze(-1)) * self.residual_scale).clone()
                # delta_sigma_v = (torch.tanh(self.net_sigma_v._model((features - scaler_mean_sigma_v) / scaler_std_sigma_v).squeeze(-1)) * self.residual_scale).clone()

                # [S/S0, v_t, r, t, z] 这些特征本身就自带良好的数值范围（都在 0～1 或标准正态分布附近），在 SDE 循环中强行做 Scaler 往往弊大于利
                delta_sigma_S = torch.tanh(self.net_sigma_S._model(features).squeeze(-1)) * self.residual_scale
                delta_mu_v = torch.tanh(self.net_mu_v._model(features).squeeze(-1)) * self.residual_scale
                delta_sigma_v = torch.tanh(self.net_sigma_v._model(features).squeeze(-1)) * self.residual_scale
                
                # 2. Heston 基准项
                if self.heston_params is not None:
                    # 标的漂移基准：μ_S_base = r
                    mu_S_base = torch.full((n_paths,), r, dtype=torch.float32)
                    
                    # 标的扩散基准：σ_S_base = √v_t
                    sigma_S_base = torch.sqrt(v_t_safe)
                    
                    # 波动率漂移基准：μ_v_base = κ(θ - v_t)
                    mu_v_base = self.heston_params.kappa * (self.heston_params.theta - v_t_safe)
                    
                    # 波动率扩散基准：σ_v_base = σ_heston·√v_t
                    # sigma_v_base = self.heston_params.sigma * torch.sqrt(v_t_safe)
                    sigma_v_base = torch.full((n_paths,), self.heston_params.sigma, dtype=torch.float32, device=v_t_safe.device)
                else:
                    # 无基准时，使用零基准（退化为纯神经网络）
                    mu_S_base = torch.zeros(n_paths, dtype=torch.float32)
                    sigma_S_base = torch.zeros(n_paths, dtype=torch.float32)
                    mu_v_base = torch.zeros(n_paths, dtype=torch.float32)
                    sigma_v_base = torch.zeros(n_paths, dtype=torch.float32)
                
                # 3. 基准 + 残差
                mu_S = mu_S_base + delta_mu_S
                sigma_S = torch.clamp(sigma_S_base + delta_sigma_S, min=1e-6, max=5.0)
                mu_v = mu_v_base + delta_mu_v
                sigma_v = torch.clamp(sigma_v_base + delta_sigma_v, min=1e-6, max=5.0)
                
                # 获取预生成的随机数（步骤3：修复布朗运动切片产生的视图）
                dW_S = all_dW_S[i].clone()
                dW_v = (rho * dW_S + rho_complement * all_dW_v_indep[i]).clone()
                
                # Euler 更新（步骤4：固化每步的资产状态）
                # 使用 softplus 替代 clamp 以保留梯度（v_t 必须为正）
                S_t_new = S_t + mu_S * S_t * dt + sigma_S * S_t * sqrt_dt * dW_S
                S_t = torch.clamp(S_t_new, min=1.0, max=1e6).clone()
                
                # 对于 v_t，使用 softplus 保证非负性且保留梯度
                v_t_raw = v_t_safe + mu_v * dt + sigma_v * torch.sqrt(v_t_safe) * sqrt_dt * dW_v
                # softplus(x) = log(1 + exp(x))，当 x > 0 时近似 x，当 x < 0 时平滑衰减到 0
                # 为避免过大，先做一次软截断
                beta = 10.0  # 控制 softplus 的锐度
                v_t = (torch.nn.functional.softplus(v_t_raw * beta, beta=beta) / beta).clone()
            
            # 返回最终状态（训练时只需要S_T）
            # 用None填充v_paths，因为训练时不需要完整路径
            return S_t, v_t
        else:
            # 推理模式：使用numpy
            sqrt_dt = np.sqrt(dt)
            
            # 设置随机种子
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            # 任务 3：引入对偶变量法 (Antithetic Variates) 降低方差
            half_paths = n_paths // 2
            n_paths = half_paths * 2  # 强制偶数
            
            # 更新路径数组为偶数路径
            S_paths = np.zeros((n_paths, self.n_steps + 1))
            v_paths = np.zeros((n_paths, self.n_steps + 1))
            S_paths[:, 0] = S0
            v_paths[:, 0] = v0
            
            # Euler-Maruyama 迭代
            for i in range(self.n_steps):
                t = i * dt
                S_t = S_paths[:, i]
                v_t = np.maximum(v_paths[:, i], 1e-6)
                
                # 构造输入特征（任务 2：使用 Moneyness 归一化 S_t）
                features = np.column_stack([
                    S_t / S0,  # Moneyness 归一化，避免 tanh 饱和
                    v_t,
                    np.full(n_paths, r),
                    np.full(n_paths, t),
                    np.tile(z, (n_paths, 1))
                ])
                
                # 1. 神经网络预测残差（显式 tanh 包裹输出）
                # 任务 2：强制 delta_mu_S = 0 以满足风险中性定价
                delta_mu_S = np.zeros(n_paths)
                # 任务 1：显式 tanh 包裹模型输出
                delta_sigma_S = np.tanh(self.net_sigma_S.predict(features).ravel()) * self.residual_scale
                delta_mu_v = np.tanh(self.net_mu_v.predict(features).ravel()) * self.residual_scale
                delta_sigma_v = np.tanh(self.net_sigma_v.predict(features).ravel()) * self.residual_scale
                
                # 2. Heston 基准
                if self.heston_params is not None:
                    mu_S_base = np.full(n_paths, r)
                    sigma_S_base = np.sqrt(v_t)
                    mu_v_base = self.heston_params.kappa * (self.heston_params.theta - v_t)
                    # sigma_v_base = self.heston_params.sigma * np.sqrt(v_t)
                    sigma_v_base = np.full(n_paths, self.heston_params.sigma)
                else:
                    mu_S_base = np.zeros(n_paths)
                    sigma_S_base = np.zeros(n_paths)
                    mu_v_base = np.zeros(n_paths)
                    sigma_v_base = np.zeros(n_paths)
                
                # 3. 基准 + 残差
                mu_S = mu_S_base + delta_mu_S
                sigma_S = np.clip(sigma_S_base + delta_sigma_S, 1e-6, 5.0)
                mu_v = mu_v_base + delta_mu_v
                sigma_v = np.clip(sigma_v_base + delta_sigma_v, 1e-6, 5.0)
                
                # 生成布朗运动增量（任务 3：对偶变量法）
                Z_S = np.random.randn(half_paths)
                dW_S = np.concatenate([Z_S, -Z_S])
                Z_v = np.random.randn(half_paths)
                dW_v_indep = np.concatenate([Z_v, -Z_v])
                dW_v = rho * dW_S + np.sqrt(1 - rho**2) * dW_v_indep
                
                # Euler 更新
                S_paths[:, i+1] = S_t + mu_S * S_t * dt + sigma_S * S_t * sqrt_dt * dW_S
                S_paths[:, i+1] = np.maximum(S_paths[:, i+1], 1e-6)
                
                v_paths[:, i+1] = v_t + mu_v * dt + sigma_v * np.sqrt(v_t) * sqrt_dt * dW_v
                v_paths[:, i+1] = np.maximum(v_paths[:, i+1], 1e-6)
            
            return S_paths, v_paths
    
    def price_option(
        self,
        S0: float,
        v0: float,
        K: float,
        r: float,
        T: float,
        z: np.ndarray,
        option_type: str = 'call',
        n_paths: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        蒙特卡洛期权定价
        
        Parameters:
        -----------
        S0, v0 : 初始标的价格和波动率方差
        K : 行权价
        r : 无风险利率
        T : 到期时间
        z : 情绪因子
        option_type : 'call' 或 'put'
        n_paths : 路径数（None 则使用 self.n_paths）
        
        Returns:
        --------
        price : 期权价格
        std_error : 标准误差
        """
        # 模拟路径
        S_paths, v_paths = self.simulate_paths(S0, v0, r, T, z, n_paths)
        
        # 计算 payoff
        S_T = S_paths[:, -1]
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # 贴现
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(len(payoffs))
        
        return price, std_error
    
    def fit(
        self,
        df_train: pd.DataFrame,
        sentiment_dict: Dict[str, np.ndarray],
        epochs: int = 50,
        batch_size: int = 64,  # 较小 batch size（因为蒙特卡洛成本高）
        lr: float = 1e-3,
        n_paths_train: int = 1000,  # 训练时减少路径数
        loss_type: Optional[str] = None,  # 损失函数类型（None则使用__init__中的设置）
        verbose: bool = True,
    ) -> "NeuralSDEPricer":
        """
        训练 NSDE 模型
        
        损失函数：市场价格与蒙特卡洛定价的损失（可选MSE/MAPE/Relative MSE）
        
        Parameters:
        -----------
        df_train : 训练数据（需包含 date 列）
        sentiment_dict : {date: z} 情绪因子字典
        epochs : 训练轮数
        batch_size : 批量大小
        lr : 学习率
        n_paths_train : 训练时的蒙特卡洛路径数
        loss_type : 损失函数类型（None则使用初始化时的设置）
        verbose : 是否打印训练进度
        
        Returns:
        --------
        self
        """
        # 如果未提供 Heston 基准参数，自动校准
        if self.heston_params is None:
            if verbose:
                print("=" * 60)
                print("未提供 Heston 基准参数，开始校准...")
            from OptionPricingModel import OptionPricingModel
            calibrator = OptionPricingModel()
            self.heston_params = calibrator.calibrate_heston(
                df_train,
                max_samples=2000,
                method='least_squares'
            )
            if verbose:
                print(f"校准完成: κ={self.heston_params.kappa:.3f}, "
                      f"θ={self.heston_params.theta:.3f}, "
                      f"σ={self.heston_params.sigma:.3f}, "
                      f"ρ={self.heston_params.rho:.3f}, "
                      f"v0={self.heston_params.v0:.3f}")
                print("=" * 60)
        
        # 准备训练数据
        df = df_train.copy()
        
        # 过滤：只保留有情绪因子的日期
        df['date_str'] = df['date'].astype(str)
        df = df[df['date_str'].isin(sentiment_dict.keys())].copy()
        
        if len(df) == 0:
            raise ValueError("训练数据中没有匹配的情绪因子日期")
        
        print(f"训练样本数: {len(df)}")
        
        # 预计算每日ATM IV（如果需要）
        daily_atm_iv = None
        if self.v0_source == 'atm_iv':
            from OptionPricingModel import compute_daily_atm_iv
            daily_atm_iv = compute_daily_atm_iv(df)
            if verbose:
                print(f"✓ 已计算 {len(daily_atm_iv)} 个交易日的ATM IV")
                print(f"  ATM IV 范围: {daily_atm_iv.min():.4f} ~ {daily_atm_iv.max():.4f}")
        
        # 提取特征
        features_list = []
        targets_list = []
        
        for _, row in df.iterrows():
            date_str = row['date_str']
            z = sentiment_dict[date_str]
            
            # 根据 v0_source 选择初始方差
            if self.v0_source == 'hv_20d':
                v0 = row['hv_20d'] ** 2  # 历史波动率的平方
            elif self.v0_source == 'atm_iv':
                # 从预计算的每日ATM IV中查找
                v0 = daily_atm_iv.loc[row['date']] ** 2
            elif self.v0_source == 'iv':
                # 直接使用该期权自己的隐含波动率（可能存在轻微泄露）
                v0 = row['iv'] ** 2
            elif self.v0_source == 'heston':
                # 使用Heston校准得到的v0
                if self.heston_params is None:
                    raise ValueError("v0_source='heston' 需要提供 heston_params 或在训练时自动校准")
                v0 = self.heston_params.v0
            else:
                raise ValueError(f"不支持的 v0_source: {self.v0_source}. 支持的选项: 'hv_20d', 'atm_iv', 'iv', 'heston'")
            
            features_list.append({
                'S0': row['underlying_close'],
                'v0': v0,
                'K': row['strike_price'],
                'r': row['risk_free_rate'],
                'T': row['time_to_expire'],
                'z': z,
                'option_type': 'call' if row['call_put'] == 0 else 'put'  # 修复：0=Call, 1=Put
            })
            targets_list.append(row['close'])
        
        n = len(features_list)
        targets = np.array(targets_list)
        
        # 初始化网络的scaler（使用初始状态的特征统计）
        # 构造一些初始状态样本用于计算统计量
        init_features = []
        for feat in features_list[:min(100, len(features_list))]:  # 使用前100个样本
            init_features.append([
                1.0,  # S_t / S0 = 1.0 在 t=0 时刻（Moneyness）
                feat['v0'], feat['r'], 0.0,  # t=0时刻
                *feat['z']
            ])
        init_features = np.array(init_features, dtype=np.float32)
        
        # 为每个网络初始化scaler
        for net in [self.net_mu_S, self.net_sigma_S, self.net_mu_v, self.net_sigma_v]:
            if net._scaler_mean is None:
                net._scaler_mean = init_features.mean(axis=0)
                net._scaler_std = init_features.std(axis=0)
                net._scaler_std[net._scaler_std < 1e-8] = 1.0
        
        # 收集所有网络参数
        all_params = []
        for net in [self.net_mu_S, self.net_sigma_S, self.net_mu_v, self.net_sigma_v]:
            all_params.extend(list(net._model.parameters()))
        
        optimizer = torch.optim.Adam(all_params, lr=lr)
        self.history_ = {
            'loss': [], 
            'mae': [],
            'grad_norm_mu_S': [],
            'grad_norm_sigma_S': [],
            'grad_norm_mu_v': [],
            'grad_norm_sigma_v': [],
            'residual_stats': [],  # 网络残差统计
            'v_t_stats': [],  # v_t 边界触发统计
        }
        
        # 训练循环
        for ep in range(epochs):
            perm = np.random.permutation(n)
            total_loss = 0.0
            total_mae = 0.0
            total_samples = 0
            
            # 梯度和残差统计
            epoch_grad_norms = {'mu_S': [], 'sigma_S': [], 'mu_v': [], 'sigma_v': []}
            epoch_residuals = {'delta_sigma_S': [], 'delta_mu_v': [], 'delta_sigma_v': []}
            epoch_v_t_min = []
            
            # 使用 tqdm 显示进度
            pbar = tqdm(range(0, n, batch_size), desc=f"Epoch {ep+1}/{epochs}") if verbose else range(0, n, batch_size)
            
            for start in pbar:
                end = min(start + batch_size, n)
                idx = perm[start:end]
                
                optimizer.zero_grad()
                
                # 批量定价（使用PyTorch保持梯度）
                batch_prices = []
                for i in idx:
                    feat = features_list[i]
                    # 使用return_torch=True进行路径模拟（返回最终状态）
                    S_T, v_T = self.simulate_paths(
                        S0=feat['S0'],
                        v0=feat['v0'],
                        r=feat['r'],
                        T=feat['T'],
                        z=feat['z'],
                        n_paths=n_paths_train,
                        return_torch=True  # 训练时使用PyTorch
                    )
                    
                    # 计算payoff（S_T已经是最终价格）
                    K_tensor = torch.tensor(feat['K'], dtype=torch.float32)
                    if feat['option_type'].lower() == 'call':
                        payoffs = torch.clamp(S_T - K_tensor, min=0.0)
                    else:
                        payoffs = torch.clamp(K_tensor - S_T, min=0.0)
                    
                    # 贴现
                    discount = torch.exp(torch.tensor(-feat['r'] * feat['T'], dtype=torch.float32))
                    price = discount * torch.mean(payoffs)
                    batch_prices.append(price)
                
                # 堆叠为tensor
                pred_t = torch.stack(batch_prices)
                batch_targets = targets[idx]
                target_t = torch.tensor(batch_targets, dtype=torch.float32)
                
                # 使用可配置的损失函数
                used_loss_type = loss_type or self.loss_type
                loss = self._compute_loss(pred_t, target_t, used_loss_type)
                mae = torch.mean(torch.abs(pred_t - target_t))
                
                # 反向传播
                loss.backward()
                
                # ===== 梯度监控 =====
                # 计算每个网络的梯度范数
                def compute_grad_norm(net):
                    total_norm = 0.0
                    for p in net._model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    return np.sqrt(total_norm)
                
                epoch_grad_norms['mu_S'].append(compute_grad_norm(self.net_mu_S))
                epoch_grad_norms['sigma_S'].append(compute_grad_norm(self.net_sigma_S))
                epoch_grad_norms['mu_v'].append(compute_grad_norm(self.net_mu_v))
                epoch_grad_norms['sigma_v'].append(compute_grad_norm(self.net_sigma_v))
                
                optimizer.step()
                
                # 累加损失
                batch_len = len(idx)
                total_loss += loss.item() * batch_len
                total_mae += mae.item() * batch_len
                total_samples += batch_len
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}', 
                        'mae': f'{mae.item():.4f}',
                        'grad': f'{epoch_grad_norms["sigma_v"][-1]:.2e}'  # 显示 sigma_v 的梯度
                    })
            
            # 记录每轮平均损失和梯度统计
            ep_loss = total_loss / total_samples
            ep_mae = total_mae / total_samples
            
            self.history_['loss'].append(ep_loss)
            self.history_['mae'].append(ep_mae)
            
            # 记录梯度范数（平均值）
            self.history_['grad_norm_mu_S'].append(np.mean(epoch_grad_norms['mu_S']))
            self.history_['grad_norm_sigma_S'].append(np.mean(epoch_grad_norms['sigma_S']))
            self.history_['grad_norm_mu_v'].append(np.mean(epoch_grad_norms['mu_v']))
            self.history_['grad_norm_sigma_v'].append(np.mean(epoch_grad_norms['sigma_v']))
            
            if verbose:
                grad_info = (f"  Grad: μ_S={self.history_['grad_norm_mu_S'][-1]:.2e}, "
                           f"σ_S={self.history_['grad_norm_sigma_S'][-1]:.2e}, "
                           f"μ_v={self.history_['grad_norm_mu_v'][-1]:.2e}, "
                           f"σ_v={self.history_['grad_norm_sigma_v'][-1]:.2e}")
                print(f"  Epoch {ep+1}/{epochs}, Loss: {ep_loss:.6f}, MAE: {ep_mae:.4f}")
                print(grad_info)
        
        return self
    
    def validate_heston_baseline(
        self,
        df_sample: pd.DataFrame,
        sentiment_dict: Dict[str, np.ndarray],
        n_paths_test: int = 5000
    ) -> Dict[str, float]:
        """
        验证纯 Heston 模型的定价精度（不使用神经网络残差）
        
        这个函数用于诊断：如果纯 Heston 效果不好，说明基准参数有问题
        
        Returns:
        --------
        dict : {'mae', 'rmse', 'mape', 'mean_pred', 'mean_target'}
        """
        from OptionPricingModel import OptionPricingModel
        
        heston_model = OptionPricingModel()
        
        # 临时保存原始残差缩放因子
        original_scale = self.residual_scale
        
        # 设置残差为 0（纯 Heston）
        self.residual_scale = 0.0
        
        df = df_sample.copy()
        df['date_str'] = df['date'].astype(str)
        
        predictions = []
        targets = []
        
        print("验证纯 Heston 基准定价...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            date_str = row['date_str']
            z = sentiment_dict.get(date_str, np.zeros(self.latent_dim))
            
            price, _ = self.price_option(
                S0=row['underlying_close'],
                v0=row['hv_20d'] ** 2,
                K=row['strike_price'],
                r=row['risk_free_rate'],
                T=row['time_to_expire'],
                z=z,
                option_type='call' if row['call_put'] > 0.5 else 'put',
                n_paths=n_paths_test
            )
            
            predictions.append(price)
            targets.append(row['close'])
        
        # 恢复原始残差缩放因子
        self.residual_scale = original_scale
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mean_pred': np.mean(predictions),
            'mean_target': np.mean(targets),
            'predictions': predictions,
            'targets': targets
        }
    
    def predict_batch(
        self,
        df_test: pd.DataFrame,
        sentiment_dict: Dict[str, np.ndarray],
        n_paths_test: int = 10000,  # 测试时增加路径数
        verbose: bool = True
    ) -> np.ndarray:
        """
        批量预测测试集
        
        Parameters:
        -----------
        df_test : 测试数据
        sentiment_dict : 情绪因子字典
        n_paths_test : 测试时的路径数
        verbose : 是否显示进度
        
        Returns:
        --------
        predictions : (n_samples,) 预测价格
        """
        df = df_test.copy()
        df['date_str'] = df['date'].astype(str)
        
        # 预计算每日ATM IV（如果需要）
        daily_atm_iv = None
        if self.v0_source == 'atm_iv':
            from OptionPricingModel import compute_daily_atm_iv
            daily_atm_iv = compute_daily_atm_iv(df)
            if verbose:
                print(f"✓ 已计算 {len(daily_atm_iv)} 个交易日的ATM IV (测试集)")
        
        predictions = []
        std_errors = []
        
        iterator = tqdm(df.iterrows(), total=len(df), desc="预测中") if verbose else df.iterrows()
        
        for _, row in iterator:
            date_str = row['date_str']
            
            # 如果没有情绪因子，使用零向量
            if date_str in sentiment_dict:
                z = sentiment_dict[date_str]
            else:
                z = np.zeros(self.latent_dim)
            
            # 根据 v0_source 选择初始方差
            if self.v0_source == 'hv_20d':
                v0 = row['hv_20d'] ** 2
            elif self.v0_source == 'atm_iv':
                v0 = daily_atm_iv.loc[row['date']] ** 2
            elif self.v0_source == 'iv':
                v0 = row['iv'] ** 2
            elif self.v0_source == 'heston':
                # 使用Heston校准得到的v0
                if self.heston_params is None:
                    raise ValueError("v0_source='heston' 需要提供 heston_params")
                v0 = self.heston_params.v0
            else:
                raise ValueError(f"不支持的 v0_source: {self.v0_source}")
            
            price, std_err = self.price_option(
                S0=row['underlying_close'],
                v0=v0,
                K=row['strike_price'],
                r=row['risk_free_rate'],
                T=row['time_to_expire'],
                z=z,
                option_type='call' if row['call_put'] == 0 else 'put',  # 修复：0=Call, 1=Put
                n_paths=n_paths_test
            )
            
            predictions.append(price)
            std_errors.append(std_err)
        
        return np.array(predictions), np.array(std_errors)
    
    def analyze_diffusion_sensitivity(
        self,
        S0: float,
        v0: float,
        r: float,
        t: float,
        z_range: Tuple[float, float] = (-2, 2),
        n_points: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        分析扩散项对情绪因子的敏感性
        
        核心科学问题：情绪因子 z 如何影响 σ_S 和 σ_v？
        
        Parameters:
        -----------
        S0, v0, r, t : 状态变量
        z_range : 情绪因子范围
        n_points : 采样点数
        
        Returns:
        --------
        dict : {'z_values', 'sigma_S', 'sigma_v', 'mu_S', 'mu_v'}
        """
        # 假设 2 维情绪因子，只变化第一维
        z_values = np.linspace(z_range[0], z_range[1], n_points)
        
        sigma_S_values = []
        sigma_v_values = []
        mu_S_values = []
        mu_v_values = []
        
        for z1 in z_values:
            z = np.array([z1, 0.0])  # 第二维固定为 0
            
            # 构造输入（使用 Moneyness 归一化）
            features = np.array([[1.0, v0, r, t, z[0], z[1]]])  # S/S0 = 1.0
            
            # 预测
            mu_S = self.net_mu_S.predict(features)[0]
            sigma_S = self.net_sigma_S.predict(features)[0]
            mu_v = self.net_mu_v.predict(features)[0]
            sigma_v = self.net_sigma_v.predict(features)[0]
            
            mu_S_values.append(mu_S)
            sigma_S_values.append(sigma_S)
            mu_v_values.append(mu_v)
            sigma_v_values.append(sigma_v)
        
        return {
            'z_values': z_values,
            'sigma_S': np.array(sigma_S_values),
            'sigma_v': np.array(sigma_v_values),
            'mu_S': np.array(mu_S_values),
            'mu_v': np.array(mu_v_values)
        }
