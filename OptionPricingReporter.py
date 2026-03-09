"""
OptionPricingReporter: 学术论文辅助类

统一处理期权定价实验的可视化、LaTeX 导出与结果对比。
- 可视化：整体 / 分档（Call/Put 分离）
- LaTeX：表格输出
- 加载：从 CSV 读取已有结果并对比

Author: Version_9
"""

from typing import Optional, Union, List, Dict, Literal
import pandas as pd
import numpy as np

# 列名映射：兼容中文列名
COL_MAP = {
    "模型": "model",
    "档位": "zone",
    "样本数": "n",
    "MAPE": "MAPE(%)",
}

SCHEMA_OVERALL = ["model", "MAE", "RMSE", "MAPE(%)"]
SCHEMA_ZONE = ["option_type", "zone", "model", "n", "MAE", "RMSE", "MAPE(%)"]
SCHEMA_LOSS = ["model", "epoch", "loss"]


def _normalize_columns(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    """将中文列名转为 schema 列名"""
    df = df.copy()
    for old, new in COL_MAP.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    if "MAPE" in df.columns and "MAPE(%)" not in df.columns:
        df = df.rename(columns={"MAPE": "MAPE(%)"})
    return df


class OptionPricingReporter:
    """
    期权定价结果报告器，用于学术论文撰写辅助。
    """

    def __init__(
        self,
        metrics_cols: Optional[List[str]] = None,
        float_format: str = "%.2f",
    ):
        self.metrics_cols = metrics_cols or ["MAE", "RMSE", "MAPE(%)"]
        self.float_format = float_format
        self._df_overall: Optional[pd.DataFrame] = None
        self._df_zone: Optional[pd.DataFrame] = None
        self._df_loss: Optional[pd.DataFrame] = None

    def add_overall(
        self,
        data: Union[pd.DataFrame, List[dict]],
        stage: Optional[str] = None,
    ) -> "OptionPricingReporter":
        """添加整体指标。data 需含 model 及 MAE/RMSE/MAPE(%)，可选 stage"""
        df = pd.DataFrame(data) if isinstance(data, list) else data.copy()
        df = _normalize_columns(df, "overall")
        required = ["model"] + [c for c in self.metrics_cols if c in df.columns or "MAPE" in df.columns]
        if "MAPE" in df.columns:
            df = df.rename(columns={"MAPE": "MAPE(%)"})
        if "model" not in df.columns and "模型" in df.columns:
            df = df.rename(columns={"模型": "model"})
        if stage is not None:
            df["stage"] = stage
        if self._df_overall is None:
            self._df_overall = df
        else:
            self._df_overall = pd.concat([self._df_overall, df], ignore_index=True)
        return self

    def add_zone(
        self,
        data: Union[pd.DataFrame, List[dict]],
        stage: Optional[str] = None,
    ) -> "OptionPricingReporter":
        """添加分档指标。data 需含 option_type, zone, model, n, MAE, RMSE, MAPE(%)"""
        df = pd.DataFrame(data) if isinstance(data, list) else data.copy()
        df = _normalize_columns(df, "zone")
        if "model" not in df.columns and "模型" in df.columns:
            df = df.rename(columns={"模型": "model"})
        if "zone" not in df.columns and "档位" in df.columns:
            df = df.rename(columns={"档位": "zone"})
        if "n" not in df.columns and "样本数" in df.columns:
            df = df.rename(columns={"样本数": "n"})
        if stage is not None:
            df["stage"] = stage
        if self._df_zone is None:
            self._df_zone = df
        else:
            self._df_zone = pd.concat([self._df_zone, df], ignore_index=True)
        return self

    def add_train_loss(
        self,
        data: Union[pd.DataFrame, List[dict]],
    ) -> "OptionPricingReporter":
        """添加训练 loss 曲线数据"""
        df = pd.DataFrame(data) if isinstance(data, list) else data.copy()
        if self._df_loss is None:
            self._df_loss = df
        else:
            self._df_loss = pd.concat([self._df_loss, df], ignore_index=True)
        return self

    def load_csv(
        self,
        path: str,
        stage: Optional[str] = None,
    ) -> "OptionPricingReporter":
        """从 CSV 加载并合并。根据列推断 schema（整体/分档/训练）"""
        df = pd.read_csv(path)
        if "option_type" in df.columns and "zone" in df.columns:
            if stage is not None and "stage" not in df.columns:
                df["stage"] = stage
            self.add_zone(df)
        elif "epoch" in df.columns and "loss" in df.columns:
            self.add_train_loss(df)
        else:
            if "model" not in df.columns and "模型" in df.columns:
                df = df.rename(columns={"模型": "model"})
            if stage is not None and "stage" not in df.columns:
                df["stage"] = stage
            self.add_overall(df)
        return self

    def plot_overall(
        self,
        split_call_put: bool = False,
        figsize: tuple = (12, 4),
    ) -> None:
        """整体指标柱状图。split_call_put=True 时从分档数据聚合（需有 _df_zone）"""
        import matplotlib.pyplot as plt

        if split_call_put and self._df_zone is not None:
            agg = self._df_zone.groupby(["option_type", "model"])[self.metrics_cols].mean().reset_index()
            for opt in agg["option_type"].unique():
                sub = agg[agg["option_type"] == opt]
                models = sub["model"].tolist()
                fig, axes = plt.subplots(1, len(self.metrics_cols), figsize=figsize)
                if len(self.metrics_cols) == 1:
                    axes = [axes]
                for j, met in enumerate(self.metrics_cols):
                    if met in sub.columns:
                        axes[j].bar(models, sub[met].values)
                        axes[j].set_ylabel(met)
                        axes[j].set_title(f"{opt} {met}")
                        axes[j].tick_params(axis="x", rotation=45)
                plt.suptitle(f"整体指标（{opt}）")
                plt.tight_layout()
                plt.show()
        elif self._df_overall is not None:
            models = self._df_overall["model"].tolist()
            fig, axes = plt.subplots(1, len(self.metrics_cols), figsize=figsize)
            if len(self.metrics_cols) == 1:
                axes = [axes]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
            for j, met in enumerate(self.metrics_cols):
                if met in self._df_overall.columns:
                    vals = self._df_overall[met].values
                    axes[j].bar(models, vals, color=colors)
                    axes[j].set_ylabel(met)
                    axes[j].tick_params(axis="x", rotation=45)
            plt.tight_layout()
            plt.show()

    def plot_zone(
        self,
        split_call_put: bool = True,
        figsize: tuple = (14, 10),
        stage: Optional[str] = None,
    ) -> None:
        """分档 3×3 图。split_call_put=True：Call/Put 各一张；False：合并。stage: 仅绘制该 stage"""
        import matplotlib.pyplot as plt

        if self._df_zone is None:
            return
        df = self._df_zone
        if stage is not None and "stage" in df.columns:
            df = df[df["stage"] == stage]
        models = df["model"].unique().tolist()
        metrics = [m for m in self.metrics_cols if m in df.columns]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))

        def _plot_one(opt_df: pd.DataFrame, title: str):
            fig, axes = plt.subplots(3, 3, figsize=figsize)
            for i, zone in enumerate(["OTM", "ATM", "ITM"]):
                d = opt_df[opt_df["zone"] == zone]
                if len(d) == 0:
                    continue
                n_zone = d["n"].iloc[0] if "n" in d.columns else ""
                for j, met in enumerate(metrics):
                    ax = axes[i, j]
                    vals = d.set_index("model").reindex(models)[met].values
                    bars = ax.bar(models, vals, color=colors)
                    ax.set_ylabel(met)
                    ax.set_title(f"{zone} {met} (n={n_zone})")
                    ax.tick_params(axis="x", rotation=45)
                    for b in bars:
                        v = b.get_height()
                        if not (np.isnan(v) or np.isinf(v)):
                            ax.text(
                                b.get_x() + b.get_width() / 2, v, self.float_format % v,
                                ha="center", va="bottom", fontsize=7, rotation=45,
                            )
            plt.suptitle(title, fontsize=12)
            plt.tight_layout()
            plt.show()

        if split_call_put and "option_type" in df.columns:
            for opt in ["Call", "Put"]:
                sub = df[df["option_type"] == opt]
                if len(sub) > 0:
                    _plot_one(sub, f"{opt} 分档损失")
        else:
            _plot_one(df, "分档损失")

    def plot_train_curves(self, figsize: tuple = (10, 4)) -> None:
        """训练 loss 曲线"""
        import matplotlib.pyplot as plt

        if self._df_loss is None:
            return
        df = self._df_loss
        models = df["model"].unique().tolist()
        fig, ax = plt.subplots(figsize=figsize)
        for i, m in enumerate(models):
            sub = df[df["model"] == m].sort_values("epoch")
            ax.plot(sub["epoch"], sub["loss"], label=m)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("训练 Loss 曲线")
        plt.tight_layout()
        plt.show()

    def to_latex_overall(self, highlight_best: bool = False) -> str:
        """整体指标 LaTeX 表"""
        if self._df_overall is None:
            return ""
        df = self._df_overall.copy()
        cols = ["model"] + [c for c in self.metrics_cols if c in df.columns]
        df = df[cols]
        if "stage" in df.columns:
            df = df[["stage"] + [c for c in cols if c != "stage"]]
        try:
            return df.style.format(self.float_format, subset=cols[1:]).to_latex()
        except Exception:
            return df.to_latex(float_format=self.float_format)

    def to_latex_zone(
        self,
        highlight_best: bool = False,
    ) -> str:
        """分档 LaTeX 表，按 option_type + metric 分表"""
        if self._df_zone is None:
            return ""
        df = self._df_zone
        out = []
        for opt in ["Call", "Put"]:
            sub = df[df["option_type"] == opt] if "option_type" in df.columns else df
            if "option_type" in df.columns and len(sub) == 0:
                continue
            for met in self.metrics_cols:
                if met not in sub.columns:
                    continue
                p = sub.pivot(index="model", columns="zone", values=met)
                p = p.reindex(columns=["OTM", "ATM", "ITM"])
                caption = f"{opt} {met}"
                try:
                    latex = p.style.format(self.float_format).to_latex(caption=caption)
                except Exception:
                    latex = p.to_latex(float_format=self.float_format, caption=caption)
                out.append(f"% {caption}\n{latex}\n")
        return "\n".join(out)

    def to_latex_comparison(
        self,
        model_mapping: Optional[Dict[str, str]] = None,
    ) -> str:
        """S3 vs S4 对比表。model_mapping: {S3 模型名: S4 模型名}"""
        if self._df_zone is None or "stage" not in self._df_zone.columns:
            return ""
        df = self._df_zone
        stages = sorted(df["stage"].unique())
        if len(stages) < 2:
            return ""
        s1, s2 = stages[0], stages[1]
        d1 = df[df["stage"] == s1].copy()
        d2 = df[df["stage"] == s2].copy()
        if model_mapping:
            d1["model"] = d1["model"].map(lambda x: model_mapping.get(x, x))
        merged = d1.merge(
            d2,
            on=["option_type", "zone", "model"],
            suffixes=(f"_{s1}", f"_{s2}"),
            how="inner",
        )
        out = []
        for opt in ["Call", "Put"]:
            m = merged[merged["option_type"] == opt]
            if len(m) == 0:
                continue
            for met in self.metrics_cols:
                c1, c2 = f"{met}_{s1}", f"{met}_{s2}"
                if c1 not in m.columns or c2 not in m.columns:
                    continue
                p = m.pivot(index="model", columns="zone", values=[c1, c2])
                p = p.round(2)
                caption = f"{opt} {met} ({s1} vs {s2})"
                try:
                    latex = p.style.format(self.float_format).to_latex(caption=caption)
                except Exception:
                    latex = p.to_latex(float_format=self.float_format, caption=caption)
                out.append(f"% {caption}\n{latex}\n")
        return "\n".join(out)

    def save_csv(
        self,
        path: str,
        which: Literal["overall", "zone", "loss"] = "zone",
        stage: Optional[str] = None,
    ) -> None:
        """
        导出为统一格式 CSV，自动添加时间戳列。
        
        Args:
            path: 保存路径
            which: 导出类型 ("overall", "zone", "loss")
            stage: 仅导出该 stage 的数据（仅 zone 有效）
        """
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if which == "overall" and self._df_overall is not None:
            df = self._df_overall.copy()
            if "stage" not in df.columns:
                df["stage"] = "S3"
            df["exported_at"] = timestamp
            cols = ["stage", "model"] + [c for c in self.metrics_cols if c in df.columns] + ["exported_at"]
            df[cols].to_csv(path, index=False)
        elif which == "zone" and self._df_zone is not None:
            df = self._df_zone.copy()
            if stage is not None and "stage" in df.columns:
                df = df[df["stage"] == stage]
            df["exported_at"] = timestamp
            df.to_csv(path, index=False)
        elif which == "loss" and self._df_loss is not None:
            df = self._df_loss.copy()
            df["exported_at"] = timestamp
            df.to_csv(path, index=False)
