"""
Risk Reversal Surface VAE (INR 隐式表达架构)

从多日期的 risk reversal surface 真实观测点中学习 2 维潜在表示，
采用 Deep Sets 编码 + Implicit Decoder，坐标轴严格使用 log_moneyness (ln(K/S))。

Author: [Your Name]
Date: 2026-02-12
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union

# [FALLBACK] PyTorch 不可用时启用回退方案
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


class RiskReversalSurfaceVAE:
    """
    Risk Reversal Surface 的 VAE 模型类（INR 架构）。
    
    以每个交易日的 RR 真实观测点为样本，学习 2 维潜在表示，
    用于后续定价模型的特征输入。
    """

    def __init__(
        self,
        latent_dim: int = 2,
        point_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        beta: float = 0.01,
    ):
        """
        Parameters:
        -----------
        latent_dim : int
            潜在变量维度，默认 2
        point_hidden_dims : list, optional
            PointEncoder 单点 MLP 隐藏层维度，默认 [64, 64]
        decoder_hidden_dims : list, optional
            Implicit Decoder MLP 隐藏层维度，默认 [128, 128, 64, 32]（3-4 层 + LeakyReLU）
        beta : float
            KL 散度项权重，默认 0.01，用于 β-VAE 控制 latent 正则强度
        """
        self.latent_dim = latent_dim
        self.point_hidden_dims = point_hidden_dims or [64, 64]
        self.decoder_hidden_dims = decoder_hidden_dims or [128, 128, 64, 32]
        self.beta = beta

        self._model = None
        self._scaler_mean: Optional[np.ndarray] = None  # (3,) for [T, log_m, rr]
        self._scaler_std: Optional[np.ndarray] = None

        self.X_: Optional[List[np.ndarray]] = None  # List of (N, 3)
        self.dates_: Optional[List[str]] = None
        self._data_source: Optional[str] = None

    def describe_samples(self) -> dict:
        """
        返回当前输入数据的样本描述统计。
        需先调用 prepare_dataset() 加载数据。
        """
        if self.X_ is None or self.dates_ is None:
            raise ValueError("请先调用 prepare_dataset() 加载数据")

        all_rr = np.concatenate([x[:, 2] for x in self.X_])
        all_T = np.concatenate([x[:, 0] for x in self.X_])
        all_m = np.concatenate([x[:, 1] for x in self.X_])

        desc = {
            "数据源": self._data_source or "未记录",
            "样本数（交易日）": len(self.dates_),
            "总观测点数": sum(len(x) for x in self.X_),
            "日均观测点数": float(np.mean([len(x) for x in self.X_])),
            "日期范围": f"{self.dates_[0]} ~ {self.dates_[-1]}" if self.dates_ else "无",
            "RR 值统计": {
                "均值": float(np.mean(all_rr)),
                "标准差": float(np.std(all_rr)),
                "最小值": float(np.min(all_rr)),
                "最大值": float(np.max(all_rr)),
            },
            "time_to_expire 范围": f"[{all_T.min():.4f}, {all_T.max():.4f}]",
            "log_moneyness 范围": f"[{all_m.min():.4f}, {all_m.max():.4f}]",
        }
        return desc

    def _collect_day_points(
        self, call_df: pd.DataFrame, put_df: pd.DataFrame
    ) -> np.ndarray:
        """
        从 Call/Put 收集单日 RR 真实观测点，不做插值。
        返回 (N, 3)：[time_to_expire, log_moneyness, rr_value]
        """
        iv_threshold = 0.001
        call_df = call_df[call_df["iv"] >= iv_threshold].copy()
        put_df = put_df[put_df["iv"] >= iv_threshold].copy()

        call_key = call_df.set_index(["strike_price", "maturity_date"])
        put_key = put_df.set_index(["strike_price", "maturity_date"])

        points = []
        for idx in put_key.index:
            if idx not in call_key.index:
                continue
            put_row = put_key.loc[idx]
            call_row = call_key.loc[idx]
            rr_value = put_row["iv"] - call_row["iv"]
            T = put_row["time_to_expire"]
            log_m = put_row["log_moneyness(ln(K/S))"]
            points.append([T, log_m, rr_value])

        if len(points) < 3:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(points, dtype=np.float32)

    def prepare_dataset(
        self,
        csv_path: Optional[str] = None,
        full_df: Optional[pd.DataFrame] = None,
        dates: Optional[List[str]] = None,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        从 CSV 或 DataFrame 按日期收集 RR 真实观测点，不做插值。

        Returns:
        --------
        data_list : List[np.ndarray]
            每个元素 shape (N, 3)，列依次为 [time_to_expire, log_moneyness, rr_value]
        dates : List[str]
            对应日期列表
        """
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            self._data_source = csv_path
        elif full_df is not None:
            df = full_df.copy()
            self._data_source = "DataFrame"
        else:
            raise ValueError("需提供 csv_path 或 full_df")

        if "date" not in df.columns:
            raise ValueError("数据需含 'date' 列")

        all_dates = sorted(df["date"].dropna().unique().astype(str).tolist())
        target_dates = [d for d in (dates or all_dates) if d in all_dates]

        data_list = []
        valid_dates = []
        for date in target_dates:
            snap = df[df["date"] == date]
            call_df = snap[snap["option_type"] == "Call"]
            put_df = snap[snap["option_type"] == "Put"]
            pts = self._collect_day_points(call_df, put_df)
            if len(pts) >= 3:
                data_list.append(pts)
                valid_dates.append(date)

        self.X_ = data_list
        self.dates_ = valid_dates
        return data_list, valid_dates

    def _fit_pytorch(
        self,
        data_list: List[np.ndarray],
        epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> None:
        """PyTorch INR-VAE 训练：仅对真实观测点计算 MSE"""
        all_pts = np.concatenate(data_list, axis=0)
        self._scaler_mean = all_pts.mean(axis=0)
        self._scaler_std = all_pts.std(axis=0)
        self._scaler_std[self._scaler_std < 1e-8] = 1.0
        # 确保 RR_value (列2) 有合理缩放，避免因数值过小导致重建误差占比过低、维度坍缩
        self._scaler_std[2] = max(self._scaler_std[2], 0.05)

        self._model = _INRVAEModule(
            latent_dim=self.latent_dim,
            point_hidden_dims=self.point_hidden_dims,
            decoder_hidden_dims=self.decoder_hidden_dims,
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        for ep in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for pts in data_list:
                pts_norm = (pts - self._scaler_mean) / self._scaler_std
                x = torch.FloatTensor(pts_norm)
                optimizer.zero_grad()
                recon, mu, logvar = self._model(x)
                target = (x[:, 2:3] if x.dim() == 2 else x[:, :, 2:3]).reshape(-1, 1)
                recon_flat = recon.reshape(-1, 1)
                loss = _vae_loss(target, recon_flat, mu, logvar, beta=self.beta)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            if verbose and (ep + 1) % 20 == 0:
                print(f"  Epoch {ep+1}/{epochs}, Loss: {total_loss/n_batches:.6f}")

    def _fit_fallback(self, data_list: List[np.ndarray], **kwargs) -> None:
        """[FALLBACK] 无 PyTorch 时使用 PCA 对点集统计量编码"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X_stat = np.zeros((len(data_list), 15), dtype=np.float32)
        for i, pts in enumerate(data_list):
            for j in range(3):
                c = pts[:, j]
                X_stat[i, j*5:(j+1)*5] = [c.mean(), c.std(), c.min(), c.max(), np.median(c)]
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_stat)
        self._scaler_mean = None
        self._scaler_std = None
        self._pca = PCA(n_components=self.latent_dim)
        self._pca.fit(X_scaled)
        self._model = "pca_fallback"

    def fit(
        self,
        data_list: List[np.ndarray],
        epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> "RiskReversalSurfaceVAE":
        """训练 INR-VAE。data_list 为 prepare_dataset 返回的列表。"""
        if HAS_TORCH:
            self._fit_pytorch(data_list, epochs, lr, verbose)
        else:
            if verbose:
                print("[FALLBACK] PyTorch 未安装，使用 PCA 简化编码")
            self._fit_fallback(data_list)
        return self

    def encode(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        输入单日或多样本观测集合，返回 latent。

        Parameters:
        -----------
        X : np.ndarray (N, 3) 或 List[np.ndarray]
            列 [time_to_expire, log_moneyness, rr_value]

        Returns:
        --------
        z : np.ndarray
            shape (2,) 或 (n_samples, 2)
        """
        if self._model is None:
            raise ValueError("请先调用 fit() 训练模型")

        single = False
        if isinstance(X, np.ndarray) and X.ndim == 2:
            X = [X]
            single = True

        if self._scaler_mean is not None:
            X_norm = [(x - self._scaler_mean) / self._scaler_std for x in X]
        else:
            X_norm = X

        if HAS_TORCH and isinstance(self._model, nn.Module):
            self._model.eval()
            z_list = []
            with torch.no_grad():
                for x in X_norm:
                    t = torch.FloatTensor(x)
                    z = self._model.encode(t)
                    z_list.append(z.numpy())
            z = np.stack(z_list, axis=0)
        else:
            X_stat = np.zeros((len(X_norm), 15), dtype=np.float32)
            for i, pts in enumerate(X_norm):
                for j in range(3):
                    c = pts[:, j]
                    X_stat[i, j*5:(j+1)*5] = [c.mean(), c.std(), c.min(), c.max(), np.median(c)]
            z = self._pca.transform(
                (X_stat - self._scaler.mean_) / np.sqrt(self._scaler.var_ + 1e-8)
            )

        if single:
            return z[0]
        return z

    def decode(
        self,
        z: np.ndarray,
        T_grid: np.ndarray,
        m_grid: np.ndarray,
    ) -> np.ndarray:
        """
        输入 latent 与查询网格，返回重建 RR 曲面。

        Parameters:
        -----------
        z : np.ndarray
            shape (2,) 或 (n_samples, 2)
        T_grid : np.ndarray
            1D 时间网格
        m_grid : np.ndarray
            1D log_moneyness 网格

        Returns:
        --------
        rr_surface : np.ndarray
            shape (len(T_grid), len(m_grid)) 或 (n_samples, len(T_grid), len(m_grid))
        """
        if self._model is None:
            raise ValueError("请先调用 fit() 训练模型")

        single = False
        if z.ndim == 1:
            z = z.reshape(1, -1)
            single = True

        T_mesh, m_mesh = np.meshgrid(T_grid, m_grid, indexing="ij")
        coords = np.stack([T_mesh.ravel(), m_mesh.ravel()], axis=1)

        if HAS_TORCH and isinstance(self._model, nn.Module):
            self._model.eval()
            n_T, n_m = len(T_grid), len(m_grid)
            rr_list = []
            with torch.no_grad():
                for zi in z:
                    z_rep = np.tile(zi, (len(coords), 1))
                    inp = np.concatenate([z_rep, coords], axis=1)
                    if self._scaler_mean is not None:
                        mean_c = self._scaler_mean[:2]
                        std_c = self._scaler_std[:2]
                        inp[:, 2:4] = (inp[:, 2:4] - mean_c) / std_c
                    t = torch.FloatTensor(inp)
                    out = self._model.decode(t)
                    rr = out.numpy().reshape(n_T, n_m)
                    if self._scaler_mean is not None:
                        rr = rr * self._scaler_std[2] + self._scaler_mean[2]
                    rr_list.append(rr)
            rr_surface = np.stack(rr_list, axis=0)
        else:
            rr_surface = np.zeros((len(z), len(T_grid), len(m_grid)), dtype=np.float32)

        if single:
            return rr_surface[0]
        return rr_surface

    def save_model(self, path: str) -> None:
        """保存模型"""
        import pickle
        state = {
            "latent_dim": self.latent_dim,
            "point_hidden_dims": self.point_hidden_dims,
            "decoder_hidden_dims": self.decoder_hidden_dims,
            "beta": self.beta,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
        }
        if HAS_TORCH and isinstance(self._model, nn.Module):
            state["model_state"] = self._model.state_dict()
            state["backend"] = "pytorch"
            torch.save(state, path)
        else:
            state["backend"] = "fallback"
            state["pca"] = self._pca
            state["scaler"] = self._scaler
            with open(path, "wb") as f:
                pickle.dump(state, f)

    def load_model(self, path: str) -> "RiskReversalSurfaceVAE":
        """加载模型"""
        import pickle
        if path.endswith(".pt") and HAS_TORCH:
            state = torch.load(path, map_location="cpu", weights_only=False)
        else:
            with open(path, "rb") as f:
                state = pickle.load(f)

        self.latent_dim = state["latent_dim"]
        self.point_hidden_dims = state.get("point_hidden_dims", [64, 64])
        self.decoder_hidden_dims = state.get("decoder_hidden_dims", [128, 128, 64, 32])
        self.beta = state.get("beta", 0.01)
        self._scaler_mean = state.get("scaler_mean")
        self._scaler_std = state.get("scaler_std")

        if state.get("backend") == "pytorch" and HAS_TORCH:
            self._model = _INRVAEModule(
                latent_dim=self.latent_dim,
                point_hidden_dims=self.point_hidden_dims,
                decoder_hidden_dims=self.decoder_hidden_dims,
            )
            self._model.load_state_dict(state["model_state"])
        else:
            self._model = "pca_fallback"
            self._pca = state.get("pca")
            self._scaler = state.get("scaler")
        return self


# ============ PyTorch INR-VAE 模块 ============

def _vae_loss(
    target: "torch.Tensor",
    recon: "torch.Tensor",
    mu: "torch.Tensor",
    logvar: "torch.Tensor",
    beta: float = 0.01,
) -> "torch.Tensor":
    """VAE 损失：MSE(仅 rr_value) + beta * KL，beta 控制 latent 正则强度"""
    recon_loss = nn.functional.mse_loss(recon, target, reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.mean()
    return recon_loss + beta * kl_loss


class PointEncoder(nn.Module):
    """Deep Sets 风格：单点 MLP -> mean 聚合 -> 输出 mu, logvar"""

    def __init__(self, input_dim: int = 3, latent_dim: int = 2, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [64, 64]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        h = self.encoder(x)
        agg = h.mean(dim=1)
        return self.fc_mu(agg), self.fc_logvar(agg)


class ImplicitDecoder(nn.Module):
    """输入 (z, T, log_moneyness) 拼接，3-4 层 MLP + LeakyReLU 输出 rr_value"""

    def __init__(self, latent_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 128, 64, 32]
        input_dim = latent_dim + 2
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(0.1))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.decoder(x).squeeze(-1)


class _INRVAEModule(nn.Module):
    """INR-VAE：PointEncoder + ImplicitDecoder"""

    def __init__(
        self,
        latent_dim: int = 2,
        point_hidden_dims: List[int] = None,
        decoder_hidden_dims: List[int] = None,
    ):
        super().__init__()
        self.point_encoder = PointEncoder(
            input_dim=3,
            latent_dim=latent_dim,
            hidden_dims=point_hidden_dims or [64, 64],
        )
        self.decoder = ImplicitDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims or [128, 128, 64, 32],
        )
        self.latent_dim = latent_dim

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        if x.dim() == 2:
            x = x.unsqueeze(0)
        mu, logvar = self.point_encoder(x)
        return mu.squeeze(0) if mu.size(0) == 1 else mu

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.decoder(x)

    def forward(self, x: "torch.Tensor"):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        mu, logvar = self.point_encoder(x)
        z = self.reparameterize(mu, logvar)
        coords = x[:, :, :2]
        B, N, _ = coords.shape
        z_rep = z.unsqueeze(1).expand(-1, N, -1)
        inp = torch.cat([z_rep, coords], dim=-1)
        recon = self.decoder(inp)
        return recon, mu, logvar
