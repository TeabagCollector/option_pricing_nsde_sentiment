"""
神经随机微分方程（Neural SDE）期权定价器 - 无情绪因子版本

对照实验版本：去除情绪因子 z，验证其在动力学建模中的作用。

标的资产价格过程：
dS_t = μ_S(S_t, v_t, r, t) dt + σ_S(S_t, v_t, r, t) dW_t^S

波动率过程：
dv_t = μ_v(S_t, v_t, r, t) dt + σ_v(S_t, v_t, r, t) dW_t^v

其中 μ_S, σ_S, μ_v, σ_v 均由神经网络参数化。

Author: Version_9
Date: 2026-02-23
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


class NeuralSDEPricerNoSentiment:
    """
    神经随机微分方程期权定价器（无情绪因子版本）
    
    使用 4 个神经网络建模 Heston 过程的漂移和扩散项：
    - net_mu_S: 标的资产漂移 μ_S(S, v, r, t)
    - net_sigma_S: 标的资产扩散 σ_S(S, v, r, t)
    - net_mu_v: 波动率漂移 μ_v(S, v, r, t)
    - net_sigma_v: 波动率扩散 σ_v(S, v, r, t)
    """
    
    def __init__(
        self,
        hidden_dims: list = [32, 32],
        n_paths: int = 5000,
        n_steps: int = 50,
        rho: Optional[float] = -0.5,
        rho_source: str = 'fixed',
        heston_params: Optional[HestonParams] = None,
        residual_scale: float = 0.3,
        loss_type: str = 'mse',
        v0_source: str = 'hv_20d',
        random_state: Optional[int] = None,
    ):
        """
        Parameters:
        -----------
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
            raise ImportError("NeuralSDEPricerNoSentiment 需要 PyTorch")
        
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.rho = rho
        self.rho_source = rho_source
        self.heston_params = heston_params
        self.residual_scale = residual_scale
        self.loss_type = loss_type
        self.v0_source = v0_source
        self.random_state = random_state or SEED
        
        # 输入维度：(S, v, r, t) = 4（无情绪因子）
        input_dim = 4
        
        # 4 个神经网络（全部使用 tanh 激活）
        self.net_mu_S = FlexibleMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='tanh',
            dropout=0.0,
            random_state=self.random_state
        )
        
        self.net_sigma_S = FlexibleMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='tanh',
            dropout=0.0,
            random_state=self.random_state + 1
        )
        
        self.net_mu_v = FlexibleMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='tanh',
            dropout=0.0,
            random_state=self.random_state + 2
        )
        
        self.net_sigma_v = FlexibleMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='tanh',
            dropout=0.0,
            random_state=self.random_state + 3
        )
        
        self.history_: Optional[dict] = None
    
    @staticmethod
    def _compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
        """计算损失函数"""
        if loss_type == 'mse':
            return torch.mean((pred - target) ** 2)
        elif loss_type == 'mape':
            return torch.mean(torch.abs((pred - target) / (target + 1e-8)))
        elif loss_type == 'relative_mse':
            return torch.mean(((pred - target) / (target + 1e-8)) ** 2)
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
    
    def _get_rho(self) -> float:
        """获取当前使用的rho值"""
        if self.rho_source == 'fixed':
            if self.rho is None:
                raise ValueError("rho_source='fixed' 但 rho 参数为 None")
            return self.rho
        elif self.rho_source == 'heston':
            if self.heston_params is None:
                raise ValueError("rho_source='heston' 需要提供 heston_params")
            return self.heston_params.rho
        else:
            raise ValueError(f"不支持的 rho_source: {self.rho_source}")
    
    def simulate_paths(
        self,
        S0: float,
        v0: float,
        r: float,
        T: float,
        n_paths: Optional[int] = None,
        return_torch: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Euler-Maruyama 方法模拟资产价格路径
        
        Parameters:
        -----------
        S0 : 初始标的价格
        v0 : 初始波动率方差
        r : 无风险利率
        T : 到期时间
        n_paths : 路径数（None 则使用 self.n_paths）
        return_torch : 是否返回PyTorch tensor（用于训练）
        
        Returns:
        --------
        S_paths : (n_paths, n_steps+1) 价格路径
        v_paths : (n_paths, n_steps+1) 波动率路径
        """
        if n_paths is None:
            n_paths = self.n_paths
        
        rho = self._get_rho()
        dt = T / self.n_steps
        
        if return_torch:
            # 训练模式：使用PyTorch保持梯度
            sqrt_dt = torch.sqrt(torch.tensor(dt, dtype=torch.float32))
            
            # 对偶变量法降低方差
            half_paths = n_paths // 2
            n_paths = half_paths * 2
            Z_S = torch.randn(self.n_steps, half_paths)
            all_dW_S = torch.cat([Z_S, -Z_S], dim=1)
            Z_v = torch.randn(self.n_steps, half_paths)
            all_dW_v_indep = torch.cat([Z_v, -Z_v], dim=1)
            
            S_t = torch.full((n_paths,), S0, dtype=torch.float32)
            v_t = torch.full((n_paths,), v0, dtype=torch.float32)
            
            # 预计算scaler
            scaler_mean_mu_S = torch.tensor(self.net_mu_S._scaler_mean, dtype=torch.float32)
            scaler_std_mu_S = torch.tensor(self.net_mu_S._scaler_std, dtype=torch.float32) + 1e-8
            scaler_mean_sigma_S = torch.tensor(self.net_sigma_S._scaler_mean, dtype=torch.float32)
            scaler_std_sigma_S = torch.tensor(self.net_sigma_S._scaler_std, dtype=torch.float32) + 1e-8
            scaler_mean_mu_v = torch.tensor(self.net_mu_v._scaler_mean, dtype=torch.float32)
            scaler_std_mu_v = torch.tensor(self.net_mu_v._scaler_std, dtype=torch.float32) + 1e-8
            scaler_mean_sigma_v = torch.tensor(self.net_sigma_v._scaler_mean, dtype=torch.float32)
            scaler_std_sigma_v = torch.tensor(self.net_sigma_v._scaler_std, dtype=torch.float32) + 1e-8
            rho_complement = torch.sqrt(torch.tensor(1 - rho**2 + 1e-8))
            
            # Euler-Maruyama 迭代
            for i in range(self.n_steps):
                t = i * dt
                v_t_safe = torch.max(v_t, torch.tensor(1e-6, dtype=torch.float32))
                
                # 构造输入特征（无情绪因子）
                features = torch.stack([
                    S_t / S0,  # Moneyness 归一化
                    v_t_safe,
                    torch.full((n_paths,), r, dtype=torch.float32),
                    torch.full((n_paths,), t, dtype=torch.float32)
                ], dim=1)
                
                # 神经网络预测残差
                delta_mu_S = torch.zeros(n_paths, dtype=torch.float32)  # 风险中性定价
                delta_sigma_S = torch.tanh(self.net_sigma_S._model(features).squeeze(-1)) * self.residual_scale
                delta_mu_v = torch.tanh(self.net_mu_v._model(features).squeeze(-1)) * self.residual_scale
                delta_sigma_v = torch.tanh(self.net_sigma_v._model(features).squeeze(-1)) * self.residual_scale
                
                # Heston 基准项
                if self.heston_params is not None:
                    mu_S_base = torch.full((n_paths,), r, dtype=torch.float32)
                    sigma_S_base = torch.sqrt(v_t_safe)
                    mu_v_base = self.heston_params.kappa * (self.heston_params.theta - v_t_safe)
                    sigma_v_base = torch.full((n_paths,), self.heston_params.sigma, dtype=torch.float32, device=v_t_safe.device)
                else:
                    mu_S_base = torch.zeros(n_paths, dtype=torch.float32)
                    sigma_S_base = torch.zeros(n_paths, dtype=torch.float32)
                    mu_v_base = torch.zeros(n_paths, dtype=torch.float32)
                    sigma_v_base = torch.zeros(n_paths, dtype=torch.float32)
                
                # 基准 + 残差
                mu_S = mu_S_base + delta_mu_S
                sigma_S = torch.clamp(sigma_S_base + delta_sigma_S, min=1e-6, max=5.0)
                mu_v = mu_v_base + delta_mu_v
                sigma_v = torch.clamp(sigma_v_base + delta_sigma_v, min=1e-6, max=5.0)
                
                # 布朗运动
                dW_S = all_dW_S[i].clone()
                dW_v = (rho * dW_S + rho_complement * all_dW_v_indep[i]).clone()
                
                # Euler 更新
                S_t_new = S_t + mu_S * S_t * dt + sigma_S * S_t * sqrt_dt * dW_S
                S_t = torch.clamp(S_t_new, min=1.0, max=1e6).clone()
                
                v_t_raw = v_t_safe + mu_v * dt + sigma_v * torch.sqrt(v_t_safe) * sqrt_dt * dW_v
                beta = 10.0
                v_t = (torch.nn.functional.softplus(v_t_raw * beta, beta=beta) / beta).clone()
            
            return S_t, v_t
        else:
            # 推理模式：使用numpy
            sqrt_dt = np.sqrt(dt)
            
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            # 对偶变量法
            half_paths = n_paths // 2
            n_paths = half_paths * 2
            
            S_paths = np.zeros((n_paths, self.n_steps + 1))
            v_paths = np.zeros((n_paths, self.n_steps + 1))
            S_paths[:, 0] = S0
            v_paths[:, 0] = v0
            
            for i in range(self.n_steps):
                t = i * dt
                S_t = S_paths[:, i]
                v_t = np.maximum(v_paths[:, i], 1e-6)
                
                # 构造输入特征（无情绪因子）
                features = np.column_stack([
                    S_t / S0,
                    v_t,
                    np.full(n_paths, r),
                    np.full(n_paths, t)
                ])
                
                # 神经网络预测残差
                delta_mu_S = np.zeros(n_paths)
                delta_sigma_S = np.tanh(self.net_sigma_S.predict(features).ravel()) * self.residual_scale
                delta_mu_v = np.tanh(self.net_mu_v.predict(features).ravel()) * self.residual_scale
                delta_sigma_v = np.tanh(self.net_sigma_v.predict(features).ravel()) * self.residual_scale
                
                # Heston 基准
                if self.heston_params is not None:
                    mu_S_base = np.full(n_paths, r)
                    sigma_S_base = np.sqrt(v_t)
                    mu_v_base = self.heston_params.kappa * (self.heston_params.theta - v_t)
                    sigma_v_base = np.full(n_paths, self.heston_params.sigma)
                else:
                    mu_S_base = np.zeros(n_paths)
                    sigma_S_base = np.zeros(n_paths)
                    mu_v_base = np.zeros(n_paths)
                    sigma_v_base = np.zeros(n_paths)
                
                # 基准 + 残差
                mu_S = mu_S_base + delta_mu_S
                sigma_S = np.clip(sigma_S_base + delta_sigma_S, 1e-6, 5.0)
                mu_v = mu_v_base + delta_mu_v
                sigma_v = np.clip(sigma_v_base + delta_sigma_v, 1e-6, 5.0)
                
                # 布朗运动
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
        option_type: str = 'call',
        n_paths: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        蒙特卡洛期权定价
        
        Returns:
        --------
        price : 期权价格
        std_error : 标准误差
        """
        S_paths, v_paths = self.simulate_paths(S0, v0, r, T, n_paths)
        
        S_T = S_paths[:, -1]
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        discount = np.exp(-r * T)
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(len(payoffs))
        
        return price, std_error
    
    def fit(
        self,
        df_train: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        n_paths_train: int = 1000,
        loss_type: Optional[str] = None,
        verbose: bool = True,
    ) -> "NeuralSDEPricerNoSentiment":
        """
        训练 NSDE 模型
        
        Parameters:
        -----------
        df_train : 训练数据
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
        
        df = df_train.copy()
        print(f"训练样本数: {len(df)}")
        
        # 预计算每日ATM IV（如果需要）
        daily_atm_iv = None
        if self.v0_source == 'atm_iv':
            from OptionPricingModel import compute_daily_atm_iv
            daily_atm_iv = compute_daily_atm_iv(df)
            if verbose:
                print(f"✓ 已计算 {len(daily_atm_iv)} 个交易日的ATM IV")
        
        # 提取特征
        features_list = []
        targets_list = []
        
        for _, row in df.iterrows():
            # 根据 v0_source 选择初始方差
            if self.v0_source == 'hv_20d':
                v0 = row['hv_20d'] ** 2
            elif self.v0_source == 'atm_iv':
                v0 = daily_atm_iv.loc[row['date']] ** 2
            elif self.v0_source == 'iv':
                v0 = row['iv'] ** 2
            elif self.v0_source == 'heston':
                v0 = self.heston_params.v0
            else:
                raise ValueError(f"不支持的 v0_source: {self.v0_source}")
            
            features_list.append({
                'S0': row['underlying_close'],
                'v0': v0,
                'K': row['strike_price'],
                'r': row['risk_free_rate'],
                'T': row['time_to_expire'],
                'option_type': 'call' if row['call_put'] == 0 else 'put'
            })
            targets_list.append(row['close'])
        
        n = len(features_list)
        targets = np.array(targets_list)
        
        # 初始化网络的scaler
        init_features = []
        for feat in features_list[:min(100, len(features_list))]:
            init_features.append([
                1.0,  # S_t / S0 = 1.0
                feat['v0'], feat['r'], 0.0  # t=0
            ])
        init_features = np.array(init_features, dtype=np.float32)
        
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
        }
        
        # 训练循环
        for ep in range(epochs):
            perm = np.random.permutation(n)
            total_loss = 0.0
            total_mae = 0.0
            total_samples = 0
            
            epoch_grad_norms = {'mu_S': [], 'sigma_S': [], 'mu_v': [], 'sigma_v': []}
            
            pbar = tqdm(range(0, n, batch_size), desc=f"Epoch {ep+1}/{epochs}") if verbose else range(0, n, batch_size)
            
            for start in pbar:
                end = min(start + batch_size, n)
                idx = perm[start:end]
                
                optimizer.zero_grad()
                
                # 批量定价
                batch_prices = []
                for i in idx:
                    feat = features_list[i]
                    S_T, v_T = self.simulate_paths(
                        S0=feat['S0'],
                        v0=feat['v0'],
                        r=feat['r'],
                        T=feat['T'],
                        n_paths=n_paths_train,
                        return_torch=True
                    )
                    
                    K_tensor = torch.tensor(feat['K'], dtype=torch.float32)
                    if feat['option_type'].lower() == 'call':
                        payoffs = torch.clamp(S_T - K_tensor, min=0.0)
                    else:
                        payoffs = torch.clamp(K_tensor - S_T, min=0.0)
                    
                    discount = torch.exp(torch.tensor(-feat['r'] * feat['T'], dtype=torch.float32))
                    price = discount * torch.mean(payoffs)
                    batch_prices.append(price)
                
                pred_t = torch.stack(batch_prices)
                target_t = torch.tensor(targets[idx], dtype=torch.float32)
                
                used_loss_type = loss_type or self.loss_type
                loss = self._compute_loss(pred_t, target_t, used_loss_type)
                mae = torch.mean(torch.abs(pred_t - target_t))
                
                loss.backward()
                
                # 梯度监控
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
                
                batch_len = len(idx)
                total_loss += loss.item() * batch_len
                total_mae += mae.item() * batch_len
                total_samples += batch_len
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}', 
                        'mae': f'{mae.item():.4f}',
                        'grad': f'{epoch_grad_norms["sigma_v"][-1]:.2e}'
                    })
            
            ep_loss = total_loss / total_samples
            ep_mae = total_mae / total_samples
            
            self.history_['loss'].append(ep_loss)
            self.history_['mae'].append(ep_mae)
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
    
    def predict_batch(
        self,
        df_test: pd.DataFrame,
        n_paths_test: int = 10000,
        verbose: bool = True
    ) -> np.ndarray:
        """
        批量预测测试集
        
        Returns:
        --------
        predictions : (n_samples,) 预测价格
        std_errors : (n_samples,) 标准误差
        """
        df = df_test.copy()
        
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
            # 根据 v0_source 选择初始方差
            if self.v0_source == 'hv_20d':
                v0 = row['hv_20d'] ** 2
            elif self.v0_source == 'atm_iv':
                v0 = daily_atm_iv.loc[row['date']] ** 2
            elif self.v0_source == 'iv':
                v0 = row['iv'] ** 2
            elif self.v0_source == 'heston':
                v0 = self.heston_params.v0
            else:
                raise ValueError(f"不支持的 v0_source: {self.v0_source}")
            
            price, std_err = self.price_option(
                S0=row['underlying_close'],
                v0=v0,
                K=row['strike_price'],
                r=row['risk_free_rate'],
                T=row['time_to_expire'],
                option_type='call' if row['call_put'] == 0 else 'put',
                n_paths=n_paths_test
            )
            
            predictions.append(price)
            std_errors.append(std_err)
        
        return np.array(predictions), np.array(std_errors)
