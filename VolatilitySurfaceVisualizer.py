"""
沪深300期权波动率曲面可视化工具

功能：
1. 从 full_option_trading_data.csv 读取日度数据
2. 绘制 Call/Put 的原始散点图和插值曲面
3. 绘制 Risk Reversal Surface (市场情绪结构)

Author: [Your Name]
Date: 2026-02-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, LinearNDInterpolator
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class VolatilitySurfaceVisualizer:
    """
    沪深300期权波动率曲面可视化类
    
    主要功能:
    - 加载指定日期的期权数据
    - 绘制波动率曲面 (散点 + 插值)
    - 绘制 Risk Reversal Surface (市场情绪指标)
    """
    
    def __init__(self, csv_path: str = "full_option_trading_data.csv"):
        """
        初始化可视化器
        
        Parameters:
        -----------
        csv_path : str
            完整期权交易数据的 CSV 文件路径
        """
        self.csv_path = csv_path
        self.full_df = None
        self.snapshot_df = None
        self.call_df = None
        self.put_df = None
        self.target_date = None
        
        # 配置中文字体和绘图参数
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """配置 Matplotlib 中文显示和全局样式"""
        plt.rcParams['font.family'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def load_data(self, date: str, moneyness_filter: bool = False) -> None:
        """
        加载指定日期的期权数据
        
        Parameters:
        -----------
        date : str
            目标日期，格式为 'YYYY-MM-DD'，例如 '2024-02-05'
        moneyness_filter : bool
            是否启用虚值程度过滤 (Call >= -0.05, Put <= 0.05)
            默认 False，不过滤
        """
        print(f"正在加载数据: {self.csv_path}")
        
        # 读取完整数据
        self.full_df = pd.read_csv(self.csv_path)
        self.target_date = date
        
        # 提取指定日期的快照数据
        self.snapshot_df = self.full_df[self.full_df['date'] == date].copy()
        
        if len(self.snapshot_df) == 0:
            raise ValueError(f"日期 {date} 没有数据，请检查日期格式或数据文件")
        
        # 分离 Call 和 Put 数据
        if moneyness_filter:
            self.call_df = self.snapshot_df[
                (self.snapshot_df['option_type'] == 'Call') &
                (self.snapshot_df['log_moneyness(ln(K/S))'] >= -0.05)
            ].copy()
            
            self.put_df = self.snapshot_df[
                (self.snapshot_df['option_type'] == 'Put') &
                (self.snapshot_df['log_moneyness(ln(K/S))'] <= 0.05)
            ].copy()
        else:
            self.call_df = self.snapshot_df[
                self.snapshot_df['option_type'] == 'Call'
            ].copy()
            
            self.put_df = self.snapshot_df[
                self.snapshot_df['option_type'] == 'Put'
            ].copy()
        
        # 数据清洗: 移除异常的极小 IV 值 (< 0.001 即 0.1%)
        # 这些值通常是数据错误或极度缺乏流动性的合约
        iv_threshold = 0.001
        call_before = len(self.call_df)
        put_before = len(self.put_df)
        
        self.call_df = self.call_df[self.call_df['iv'] >= iv_threshold].copy()
        self.put_df = self.put_df[self.put_df['iv'] >= iv_threshold].copy()
        
        call_removed = call_before - len(self.call_df)
        put_removed = put_before - len(self.put_df)
        
        print(f"✓ 数据加载成功")
        print(f"  Call 数据点: {len(self.call_df)}")
        print(f"  Put 数据点: {len(self.put_df)}")
        if call_removed > 0 or put_removed > 0:
            print(f"  已移除异常数据: Call {call_removed} 个, Put {put_removed} 个")
        print(f"  到期日数量: {self.snapshot_df['maturity_date'].nunique()}")
    
    def print_diagnostics(self) -> None:
        """打印 IV 数据的诊断信息"""
        if self.call_df is None or self.put_df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        print("=" * 60)
        print("IV 数据诊断")
        print("=" * 60)
        
        # Call 数据统计
        print(f"\nCall 数据 IV 统计:")
        print(f"  数据点数: {len(self.call_df)}")
        print(f"  IV 最小值: {self.call_df['iv'].min():.6f}")
        print(f"  IV 最大值: {self.call_df['iv'].max():.6f}")
        print(f"  IV 均值: {self.call_df['iv'].mean():.6f}")
        print(f"  负值数量: {(self.call_df['iv'] < 0).sum()}")
        print(f"  NaN 数量: {self.call_df['iv'].isna().sum()}")
        
        # Put 数据统计
        print(f"\nPut 数据 IV 统计:")
        print(f"  数据点数: {len(self.put_df)}")
        print(f"  IV 最小值: {self.put_df['iv'].min():.6f}")
        print(f"  IV 最大值: {self.put_df['iv'].max():.6f}")
        print(f"  IV 均值: {self.put_df['iv'].mean():.6f}")
        print(f"  负值数量: {(self.put_df['iv'] < 0).sum()}")
        print(f"  NaN 数量: {self.put_df['iv'].isna().sum()}")
    
    @staticmethod
    def _optimize_axes(ax):
        """
        优化 3D 坐标轴：减少刻度数量并格式化小数
        
        Parameters:
        -----------
        ax : Axes3D
            三维坐标轴对象
        """
        # 限制刻度数量
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
        
        # 格式化为两位小数
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # 微调旋转和间距
        ax.tick_params(axis='x', rotation=0, pad=2)
        ax.tick_params(axis='y', rotation=30, pad=2)
        ax.tick_params(axis='z', pad=5)
    
    def plot_volatility_surfaces(
        self, 
        grid_resolution: int = 50,
        figsize: Tuple[int, int] = (18, 12),
        elev: int = 20,
        azim: int = 45
    ) -> plt.Figure:
        """
        绘制 Call 和 Put 的波动率曲面 (2x2 布局)
        
        Parameters:
        -----------
        grid_resolution : int
            插值网格分辨率，默认 50
        figsize : tuple
            图形尺寸 (宽, 高)，默认 (18, 12)
        elev : int
            视角仰角，默认 20
        azim : int
            视角方位角，默认 45
            
        Returns:
        --------
        fig : Figure
            Matplotlib 图形对象
        """
        if self.call_df is None or self.put_df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        # 创建画布（减小子图间距）
        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f'沪深300指数期权波动率结构对比 ({self.target_date})', 
            fontsize=36, fontweight='bold', y=0.98
        )
        
        # 提取数据
        x_call = self.call_df['log_moneyness(ln(K/S))'].values
        y_call = self.call_df['time_to_expire'].values
        z_call = self.call_df['iv'].values
        
        x_put = self.put_df['log_moneyness(ln(K/S))'].values
        y_put = self.put_df['time_to_expire'].values
        z_put = self.put_df['iv'].values
        
        # ============================================================
        # 第一行：原始散点图
        # ============================================================
        
        # 左上：Call 原始
        ax1 = fig.add_subplot(221, projection='3d')
        scatter1 = ax1.scatter(x_call, y_call, z_call, c=z_call, 
                              cmap='viridis', s=30, alpha=0.6)
        ax1.set_title('Call 期权 - 原始数据', fontsize=24, fontweight='bold', pad=15)
        ax1.set_xlabel('ln(K/S)', fontsize=16, labelpad=12)
        ax1.set_ylabel('Time to Maturity', fontsize=16, labelpad=12)
        ax1.set_zlabel('IV', fontsize=16, labelpad=12)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.view_init(elev=elev, azim=azim)
        self._optimize_axes(ax1)
        fig.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=10)
        
        # 右上：Put 原始
        ax2 = fig.add_subplot(222, projection='3d')
        scatter2 = ax2.scatter(x_put, y_put, z_put, c=z_put, 
                              cmap='plasma', s=30, alpha=0.6)
        ax2.set_title('Put 期权 - 原始数据', fontsize=24, fontweight='bold', pad=15)
        ax2.set_xlabel('ln(K/S)', fontsize=16, labelpad=12)
        ax2.set_ylabel('Time to Maturity', fontsize=16, labelpad=12)
        ax2.set_zlabel('IV', fontsize=16, labelpad=12)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.view_init(elev=elev, azim=azim)
        self._optimize_axes(ax2)
        fig.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=10)
        
        # ============================================================
        # 第二行：插值曲面
        # ============================================================
        
        # 左下：Call 曲面
        ax3 = fig.add_subplot(223, projection='3d')
        xi_call = np.linspace(x_call.min(), x_call.max(), grid_resolution)
        yi_call = np.linspace(y_call.min(), y_call.max(), grid_resolution)
        xi_grid_call, yi_grid_call = np.meshgrid(xi_call, yi_call)
        
        # 稳健插值策略: 先尝试三次样条,失败则降级到线性插值
        zi_grid_call = griddata((x_call, y_call), z_call, 
                                (xi_grid_call, yi_grid_call), method='cubic')
        
        # 检查三次样条是否成功
        valid_cubic_call = np.sum(~np.isnan(zi_grid_call))
        interpolation_method_call = 'cubic'
        
        if valid_cubic_call < grid_resolution * grid_resolution * 0.1:  # 少于10%有效点
            # 降级到线性插值
            zi_grid_call = griddata((x_call, y_call), z_call, 
                                    (xi_grid_call, yi_grid_call), method='linear')
            interpolation_method_call = 'linear'
        
        # 只过滤负值 (理论上 IV 不能为负)
        zi_grid_call = np.where(zi_grid_call < 0, np.nan, zi_grid_call)
        
        surf1 = ax3.plot_surface(xi_grid_call, yi_grid_call, zi_grid_call, 
                                cmap='viridis', alpha=0.85, edgecolor='none')
        ax3.scatter(x_call, y_call, z_call, c='red', s=5, alpha=0.3)
        
        ax3.set_title(f'Call 期权 - {interpolation_method_call.title()} 插值', fontsize=24, fontweight='bold', pad=15)
        ax3.set_xlabel('ln(K/S)', fontsize=16, labelpad=12)
        ax3.set_ylabel('Time to Maturity', fontsize=16, labelpad=12)
        ax3.set_zlabel('IV', fontsize=16, labelpad=12)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        ax3.view_init(elev=elev, azim=azim)
        
        valid_z_call = zi_grid_call[~np.isnan(zi_grid_call)]
        if len(valid_z_call) > 0:
            ax3.set_zlim(0, valid_z_call.max() * 1.1)
        
        self._optimize_axes(ax3)
        fig.colorbar(surf1, ax=ax3, shrink=0.5, aspect=10)
        
        # 右下：Put 曲面
        ax4 = fig.add_subplot(224, projection='3d')
        xi_put = np.linspace(x_put.min(), x_put.max(), grid_resolution)
        yi_put = np.linspace(y_put.min(), y_put.max(), grid_resolution)
        xi_grid_put, yi_grid_put = np.meshgrid(xi_put, yi_put)
        
        # 稳健插值策略: 先尝试三次样条,失败则降级到线性插值
        zi_grid_put = griddata((x_put, y_put), z_put, 
                               (xi_grid_put, yi_grid_put), method='cubic')
        
        # 检查三次样条是否成功
        valid_cubic_put = np.sum(~np.isnan(zi_grid_put))
        interpolation_method_put = 'cubic'
        
        if valid_cubic_put < grid_resolution * grid_resolution * 0.1:  # 少于10%有效点
            # 降级到线性插值
            zi_grid_put = griddata((x_put, y_put), z_put, 
                                   (xi_grid_put, yi_grid_put), method='linear')
            interpolation_method_put = 'linear'
        
        # 只过滤负值 (理论上 IV 不能为负)
        zi_grid_put = np.where(zi_grid_put < 0, np.nan, zi_grid_put)
        
        surf2 = ax4.plot_surface(xi_grid_put, yi_grid_put, zi_grid_put, 
                                cmap='plasma', alpha=0.85, edgecolor='none')
        ax4.scatter(x_put, y_put, z_put, c='red', s=5, alpha=0.3)
        
        ax4.set_title(f'Put 期权 - {interpolation_method_put.title()} 插值', fontsize=24, fontweight='bold', pad=15)
        ax4.set_xlabel('ln(K/S)', fontsize=16, labelpad=12)
        ax4.set_ylabel('Time to Maturity', fontsize=16, labelpad=12)
        ax4.set_zlabel('IV', fontsize=16, labelpad=12)
        ax4.tick_params(axis='both', which='major', labelsize=14)
        ax4.view_init(elev=elev, azim=azim)
        
        valid_z_put = zi_grid_put[~np.isnan(zi_grid_put)]
        if len(valid_z_put) > 0:
            ax4.set_zlim(0, valid_z_put.max() * 1.1)
        
        self._optimize_axes(ax4)
        fig.colorbar(surf2, ax=ax4, shrink=0.5, aspect=10)
        
        # 收紧子图间距，使四幅图更紧凑
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1.5, w_pad=1.0)
        
        return fig
    
    def plot_risk_reversal_surface(
        self,
        dist_max: float = 0.15,
        grid_resolution: int = 50,
        figsize: Tuple[int, int] = (12, 8),
        elev: int = 20,
        azim: int = -60
    ) -> plt.Figure:
        """
        绘制 Risk Reversal Surface (市场情绪结构)
        
        核心逻辑：
        - 在相同虚值程度下，比较 Put IV 和 Call IV 的差异
        - Spread > 0 表示恐慌情绪 (Put 更贵)
        - Spread < 0 表示贪婪情绪 (Call 更贵)
        
        Parameters:
        -----------
        dist_max : float
            最大虚值程度 (OTM Depth)，默认 0.15 (15%)
        grid_resolution : int
            插值网格分辨率，默认 50
        figsize : tuple
            图形尺寸，默认 (12, 8) - 增大以适应更大字体
        elev : int
            视角仰角，默认 20
        azim : int
            视角方位角，默认 -60
            
        Returns:
        --------
        fig : Figure
            Matplotlib 图形对象
        """
        if self.call_df is None or self.put_df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        # 提取数据
        x_call = self.call_df['log_moneyness(ln(K/S))'].values
        y_call = self.call_df['time_to_expire'].values
        z_call = self.call_df['iv'].values
        
        x_put = self.put_df['log_moneyness(ln(K/S))'].values
        y_put = self.put_df['time_to_expire'].values
        z_put = self.put_df['iv'].values
        
        # 构建插值器
        interp_put = LinearNDInterpolator(list(zip(x_put, y_put)), z_put)
        interp_call = LinearNDInterpolator(list(zip(x_call, y_call)), z_call)
        
        # 定义新的网格：绝对虚值程度
        dist_steps = np.linspace(0, dist_max, grid_resolution)
        time_steps = np.linspace(
            min(y_call.min(), y_put.min()), 
            max(y_call.max(), y_put.max()), 
            grid_resolution
        )
        
        dist_grid, time_grid = np.meshgrid(dist_steps, time_steps)
        
        # 计算情绪价差
        # Put 取左边 (-dist), Call 取右边 (+dist)
        iv_put_side = interp_put(-dist_grid, time_grid)
        iv_call_side = interp_call(dist_grid, time_grid)
        
        # 核心公式：同等虚值下的 Put IV - Call IV
        sentiment_spread = iv_put_side - iv_call_side
        
        # 绘图
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制曲面 (coolwarm 色谱：红色=恐慌，蓝色=贪婪)
        surf = ax.plot_surface(dist_grid, time_grid, sentiment_spread, 
                              cmap='coolwarm', edgecolor='none', alpha=0.9)
        
        # 添加零平面作为参考基准
        ax.plot_surface(dist_grid, time_grid, np.zeros_like(dist_grid), 
                       color='gray', alpha=0.3)
        
        ax.set_xlabel('OTM Depth (|ln(K/S)|)', fontsize=18, labelpad=15)
        ax.set_ylabel('Time to Maturity', fontsize=18, labelpad=15)
        ax.set_zlabel('Skew (Put IV - Call IV)', fontsize=18, labelpad=15)
        ax.set_title(
            f'市场情绪结构: 风险逆转 (Risk Reversal)\n{self.target_date}', 
            fontsize=24, fontweight='bold', pad=20
        )
        
        # 增大刻度标签字体
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        ax.view_init(elev=elev, azim=azim)
        self._optimize_axes(ax)
        
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, 
                           label='Spread > 0 implies Fear')
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Spread > 0 implies Fear', fontsize=16)
        
        plt.tight_layout()
        
        return fig
    
    def get_risk_reversal_surface_data(
        self,
        dist_max: float = 0.15,
        grid_resolution: int = 50,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回 Risk Reversal Surface 的数值数据（不绘图），供 VAE 等下游使用。
        
        支持全局固定网格：传入 t_min、t_max 时使用统一时间范围，保证多日期度量一致。
        
        Parameters:
        -----------
        dist_max : float
            最大虚值程度 (OTM Depth)，默认 0.15
        grid_resolution : int
            网格分辨率，默认 50
        t_min : float, optional
            时间轴最小值。若为 None 则用当日 time_to_expire 的 min
        t_max : float, optional
            时间轴最大值。若为 None 则用当日 time_to_expire 的 max
            
        Returns:
        --------
        dist_grid : np.ndarray
            OTM Depth 网格，shape (grid_resolution, grid_resolution)
        time_grid : np.ndarray
            到期时间网格，shape (grid_resolution, grid_resolution)
        sentiment_spread : np.ndarray
            Put IV - Call IV，shape (grid_resolution, grid_resolution)
            NaN 已用 0 填充（边界外视为中性）
        """
        if self.call_df is None or self.put_df is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        x_call = self.call_df['log_moneyness(ln(K/S))'].values
        y_call = self.call_df['time_to_expire'].values
        z_call = self.call_df['iv'].values
        
        x_put = self.put_df['log_moneyness(ln(K/S))'].values
        y_put = self.put_df['time_to_expire'].values
        z_put = self.put_df['iv'].values
        
        interp_put = LinearNDInterpolator(list(zip(x_put, y_put)), z_put)
        interp_call = LinearNDInterpolator(list(zip(x_call, y_call)), z_call)
        
        dist_steps = np.linspace(0, dist_max, grid_resolution)
        if t_min is not None and t_max is not None:
            time_steps = np.linspace(t_min, t_max, grid_resolution)
        else:
            time_steps = np.linspace(
                min(y_call.min(), y_put.min()),
                max(y_call.max(), y_put.max()),
                grid_resolution
            )
        
        dist_grid, time_grid = np.meshgrid(dist_steps, time_steps)
        iv_put_side = interp_put(-dist_grid, time_grid)
        iv_call_side = interp_call(dist_grid, time_grid)
        sentiment_spread = iv_put_side - iv_call_side
        
        # NaN 填充：边界外用 0（IV 差在边界外视为中性）
        sentiment_spread = np.nan_to_num(sentiment_spread, nan=0.0)
        
        return dist_grid, time_grid, sentiment_spread
    
    def plot_all(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        一键绘制所有图表
        
        Parameters:
        -----------
        save_path : str, optional
            保存路径前缀，例如 'outputs/2024-02-05'
            会自动添加后缀 '_volatility.png' 和 '_risk_reversal.png'
        show : bool
            是否显示图形，默认 True
            
        Returns:
        --------
        fig_vol, fig_rr : tuple
            (波动率曲面图, 风险逆转图)
        """
        print(f"\n{'=' * 60}")
        print(f"开始绘制 {self.target_date} 的完整分析图表")
        print(f"{'=' * 60}\n")
        
        # 绘制波动率曲面
        print("1/2 正在绘制波动率曲面...")
        fig_vol = self.plot_volatility_surfaces()
        
        # 绘制风险逆转曲面
        print("2/2 正在绘制风险逆转曲面...")
        fig_rr = self.plot_risk_reversal_surface()
        
        # 保存图片
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            vol_path = f"{save_path}_volatility.png"
            rr_path = f"{save_path}_risk_reversal.png"
            
            fig_vol.savefig(vol_path, dpi=150, bbox_inches='tight')
            fig_rr.savefig(rr_path, dpi=150, bbox_inches='tight')
            
            print(f"\n✓ 图片已保存:")
            print(f"  - {vol_path}")
            print(f"  - {rr_path}")
        
        if show:
            plt.show()
        
        return fig_vol, fig_rr


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 创建可视化器实例
    visualizer = VolatilitySurfaceVisualizer(
        csv_path="full_option_trading_data.csv"
    )
    
    # 加载指定日期的数据
    visualizer.load_data(
        date="2024-02-05",
        moneyness_filter=False  # 不过滤虚值程度
    )
    
    # 打印数据诊断信息
    visualizer.print_diagnostics()
    
    # 一键绘制所有图表
    fig_vol, fig_rr = visualizer.plot_all(
        save_path="outputs/2024-02-05",  # 保存路径前缀
        show=True  # 显示图形
    )
    
    # 也可以单独调用各个方法
    # fig_vol = visualizer.plot_volatility_surfaces(grid_resolution=50)
    # fig_rr = visualizer.plot_risk_reversal_surface(dist_max=0.15)
