"""
可配置前馈神经网络，用于期权定价等回归任务。

支持自定义层数、隐藏层维度、激活函数、Dropout、自定义损失函数。
不涉及 RNN/LSTM。

Author: Version_9
"""

from typing import Optional, List, Union, Callable, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


def _get_activation(activation: Union[str, type]) -> nn.Module:
    """将激活函数名或类转换为 nn.Module"""
    if activation is None:
        return nn.Identity()
    if isinstance(activation, str):
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        cls = act_map.get(activation.lower())
        if cls is None:
            raise ValueError(f"未知激活函数: {activation}，支持: {list(act_map.keys())}")
        return cls()
    if isinstance(activation, nn.Module):
        return activation
    if callable(activation) and hasattr(activation, "__mro__"):
        return activation()
    raise ValueError(f"activation 应为 str、nn.Module 或可实例化类: {type(activation)}")


def create_combined_loss(alpha: float = 1.0, beta: float = 1.0):
    """
    创建「价格绝对偏差 + Heston 定价偏差」的损失函数。

    L = α * mean(|pred - y_market|) + β * mean(|pred - y_heston|)

    Returns:
        loss_fn(pred, y_true, y_heston=None, **kwargs) -> Tensor
        训练时需传入 y_heston（与 y_true 同 shape 的 Tensor）
    """
    def loss_fn(pred, y_true, y_heston=None, **kwargs):
        if not HAS_TORCH:
            raise ImportError("需要 PyTorch")
        pred = pred.reshape(-1)
        y_true = y_true.reshape(-1)
        price_dev = torch.mean(torch.abs(pred - y_true))
        if y_heston is not None:
            y_heston = y_heston.reshape(-1).to(pred.device)
            heston_dev = torch.mean(torch.abs(pred - y_heston))
            return alpha * price_dev + beta * heston_dev
        return alpha * price_dev
    return loss_fn


class FlexibleMLP:
    """
    可配置前馈神经网络，支持自定义损失函数。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
        activation: Union[str, nn.Module, type] = "relu",
        dropout: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        """
        Parameters:
        -----------
        input_dim : 输入维度（如 BS 参数 S,K,r,T,sigma,call_put 则 input_dim=6）
        hidden_dims : 隐藏层维度列表，如 [64, 64, 32]，默认 [64, 32]
        output_dim : 输出维度，默认 1（预测价格）
        activation : 激活函数，"relu"/"tanh"/"leaky_relu"/"elu" 或 nn.Module
        dropout : 0~1 或 None，每隐藏层后是否加 Dropout
        random_state : 随机种子
        """
        if not HAS_TORCH:
            raise ImportError("FlexibleMLP 需要 PyTorch，请安装: pip install torch")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.output_dim = output_dim
        self._activation = activation
        self.dropout = dropout
        self.random_state = random_state

        self._model: Optional[nn.Module] = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self.history_: Optional[dict] = None  # {"loss": [epoch_loss, ...]}

        self._build_net()

    def _build_net(self) -> None:
        """动态构建 nn.Sequential"""
        act = _get_activation(self._activation)
        layers = []
        dims = [self.input_dim] + list(self.hidden_dims) + [self.output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
                if self.dropout is not None and self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))

        self._model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier 初始化"""
        for m in self._model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _to_tensor(self, x: np.ndarray, dtype=torch.float32) -> "torch.Tensor":
        return torch.as_tensor(x, dtype=dtype)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn: Optional[Callable[..., Any]] = None,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = True,
        **kwargs,
    ) -> "FlexibleMLP":
        """
        训练模型。

        Parameters:
        -----------
        X : (n_samples, input_dim) 特征
        y : (n_samples,) 目标（市场价）
        loss_fn : 自定义损失，签名为 (pred, y_true, **kwargs) -> Tensor；
                  默认 None 使用 MSE
        epochs, batch_size, lr : 训练超参
        **kwargs : 传给 loss_fn 的额外参数，如 y_heston
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # 标准化
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std = X.std(axis=0)
        self._scaler_std[self._scaler_std < 1e-8] = 1.0
        X_norm = (X - self._scaler_mean) / self._scaler_std

        if loss_fn is None:
            def loss_fn(p, yt, **kw):
                return torch.mean((p.reshape(-1) - yt.reshape(-1)) ** 2)

        # 将 kwargs 中的 numpy 转为可分批的
        extra_tensors = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                extra_tensors[k] = torch.as_tensor(v, dtype=torch.float32)
            elif isinstance(v, torch.Tensor):
                extra_tensors[k] = v

        n = len(X_norm)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self.history_ = {"loss": []}

        for ep in range(epochs):
            perm = np.random.permutation(n)
            total_loss = 0.0
            total_samples = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                x_batch = self._to_tensor(X_norm[idx])
                y_batch = self._to_tensor(y[idx])

                batch_kw = {}
                for k, v in extra_tensors.items():
                    batch_kw[k] = v[idx]

                optimizer.zero_grad()
                pred = self._model(x_batch)
                loss = loss_fn(pred, y_batch, **batch_kw)
                loss.backward()
                optimizer.step()
                # 累加该 batch 的总损失（loss.item() 是 batch 内的平均损失）
                total_loss += loss.item() * len(idx)
                total_samples += len(idx)

            # 计算每个样本的平均损失
            ep_loss = total_loss / total_samples
            self.history_["loss"].append(ep_loss)
            if verbose and (ep + 1) % 20 == 0:
                print(f"  Epoch {ep+1}/{epochs}, Loss: {ep_loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测，返回 (n_samples,) numpy 数组"""
        X = np.asarray(X, dtype=np.float64)
        if self._scaler_mean is not None:
            X = (X - self._scaler_mean) / self._scaler_std
        self._model.eval()
        with torch.no_grad():
            x_t = self._to_tensor(X)
            out = self._model(x_t)
        pred = out.cpu().numpy().ravel()
        return np.maximum(pred, 0.0)  # 价格非负
