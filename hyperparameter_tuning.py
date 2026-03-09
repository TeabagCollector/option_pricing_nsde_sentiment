"""
超参数调优实验框架

设计目标：
1. 轻松配置多组超参数
2. 顺序训练（带进度监控和时间估算）
3. 自动保存结果和可视化对比

注意：由于PyTorch模型无法在进程间序列化，使用顺序训练而非并行。
这样更稳定可靠，且能实时查看每个配置的训练进度。

Author: Version_9
Date: 2026-02-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import pickle
import hashlib
import os
import tempfile
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from NeuralSDEPricer import NeuralSDEPricer
from OptionPricingModel import compute_metrics, set_seed

plt.rcParams['font.family'] = ['Arial Unicode MS']


# ===================== 配置缓存管理函数 =====================

def compute_config_hash(config: 'HyperparameterConfig') -> str:
    """
    计算配置的哈希值用于唯一标识
    
    基于关键训练参数生成MD5哈希，不包括辅助参数（如verbose, save_models等）
    
    Parameters:
    -----------
    config : HyperparameterConfig
        超参数配置对象
    
    Returns:
    --------
    str : 16位MD5哈希字符串
    """
    # 提取关键参数
    key_params = {
        'name': config.name,
        'latent_dim': config.latent_dim,
        'hidden_dims': tuple(config.hidden_dims),  # list转tuple以便hash
        'residual_scale': config.residual_scale,
        'loss_type': config.loss_type,      # 新增
        'v0_source': config.v0_source,      # 新增
        'rho_source': config.rho_source,    # 新增
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'lr': config.lr,
        'n_paths_train': config.n_paths_train,
        'n_steps': config.n_steps,
        'rho': config.rho,
        'random_state': config.random_state
    }
    
    # 生成哈希
    config_str = json.dumps(key_params, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]  # 使用前16位


def load_experiment_history(history_path: str) -> dict:
    """
    加载实验历史记录
    
    Parameters:
    -----------
    history_path : str
        历史记录文件路径
    
    Returns:
    --------
    dict : 实验历史记录字典
    """
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  加载历史记录失败: {e}")
            print("   创建新的历史记录文件")
    
    # 返回空结构
    return {
        'metadata': {
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_experiments': 0,
            'train_data_file': None,
            'test_data_file': None,
            'train_size': 0,
            'test_size': 0
        },
        'experiments': {}
    }


def save_experiment_history(history: dict, history_path: str):
    """
    保存实验历史记录（原子性写入）
    
    Parameters:
    -----------
    history : dict
        实验历史记录
    history_path : str
        保存路径
    """
    # 更新元数据
    history['metadata']['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history['metadata']['total_experiments'] = len(history['experiments'])
    
    # 原子性写入：先写临时文件，再重命名
    dir_name = os.path.dirname(history_path) or '.'
    fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix='.json.tmp')
    
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        # 重命名（原子操作）
        os.replace(temp_path, history_path)
    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def check_cached_result(
    config_hash: str,
    history: dict,
    train_file: str,
    test_file: str
) -> Optional['ExperimentResult']:
    """
    检查是否存在缓存的结果
    
    Parameters:
    -----------
    config_hash : str
        配置哈希值
    history : dict
        实验历史记录
    train_file : str
        训练数据文件路径
    test_file : str
        测试数据文件路径
    
    Returns:
    --------
    ExperimentResult or None : 缓存的结果或None
    """
    if config_hash not in history['experiments']:
        return None
    
    exp = history['experiments'][config_hash]
    
    # 检查数据文件是否匹配
    exp_train_file = exp.get('data_files', {}).get('train')
    exp_test_file = exp.get('data_files', {}).get('test')
    
    if exp_train_file != train_file or exp_test_file != test_file:
        return None
    
    # 重建 ExperimentResult 对象
    from hyperparameter_tuning import HyperparameterConfig, ExperimentResult
    
    config = HyperparameterConfig.from_dict(exp['config'])
    
    result = ExperimentResult(
        config=config,
        metrics=exp['metrics'],
        predictions=np.array(exp.get('predictions', [])),
        targets=np.array(exp.get('targets', [])),
        training_history=exp.get('training_history'),
        model_path=exp.get('model_path'),
        training_time=exp.get('training_time', 0)
    )
    
    return result


# ===================== 原有代码 =====================

class HyperparameterConfig:
    """超参数配置类"""
    
    def __init__(
        self,
        name: str,
        latent_dim: int = 2,
        hidden_dims: List[int] = [32, 32],
        n_paths: int = 5000,
        n_steps: int = 50,
        residual_scale: float = 0.3,
        loss_type: str = 'mse',        # 损失函数类型
        v0_source: str = 'hv_20d',     # 初始方差来源
        rho_source: str = 'fixed',     # 新增：rho来源
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        n_paths_train: int = 1000,
        n_paths_test: int = 10000,
        rho: Optional[float] = -0.5,   # 改为Optional
        random_state: Optional[int] = None
    ):
        """
        Parameters:
        -----------
        name : 配置名称（用于标识）
        latent_dim : 情绪因子维度
        hidden_dims : 隐藏层维度列表
        n_paths : 推理时蒙特卡洛路径数
        n_steps : 时间离散步数
        residual_scale : 残差缩放因子
        loss_type : 损失函数类型 ('mse', 'mape', 'relative_mse')
        v0_source : 初始方差来源 ('hv_20d', 'atm_iv', 'iv', 'heston')
        rho_source : rho来源 ('fixed', 'heston')
        epochs : 训练轮数
        batch_size : 批量大小
        lr : 学习率
        n_paths_train : 训练时路径数
        n_paths_test : 测试时路径数
        rho : 相关系数（当rho_source='fixed'时使用）
        random_state : 随机种子
        """
        self.name = name
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.residual_scale = residual_scale
        self.loss_type = loss_type
        self.v0_source = v0_source
        self.rho_source = rho_source
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_paths_train = n_paths_train
        self.n_paths_test = n_paths_test
        self.rho = rho
        self.random_state = random_state
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'name': self.name,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'n_paths': self.n_paths,
            'n_steps': self.n_steps,
            'residual_scale': self.residual_scale,
            'loss_type': self.loss_type,
            'v0_source': self.v0_source,
            'rho_source': self.rho_source,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'n_paths_train': self.n_paths_train,
            'n_paths_test': self.n_paths_test,
            'rho': self.rho,
            'random_state': self.random_state
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'HyperparameterConfig':
        """从字典创建"""
        return cls(**config_dict)


class ExperimentResult:
    """实验结果类"""
    
    def __init__(
        self,
        config: HyperparameterConfig,
        metrics: Dict[str, float],
        predictions: np.ndarray,
        targets: np.ndarray,
        training_history: Optional[dict] = None,
        model_path: Optional[str] = None,
        training_time: Optional[float] = None
    ):
        self.config = config
        self.metrics = metrics
        self.predictions = predictions
        self.targets = targets
        self.training_history = training_history
        self.model_path = model_path
        self.training_time = training_time
    
    def to_dict(self) -> dict:
        """转换为字典（用于保存）"""
        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics,
            'predictions': self.predictions.tolist(),
            'targets': self.targets.tolist(),
            'training_history': self.training_history,
            'model_path': self.model_path,
            'training_time': self.training_time
        }


def train_single_config(
    config: HyperparameterConfig,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    sentiment_dict: Dict[str, np.ndarray],
    save_dir: Path,
    verbose: bool = True
) -> ExperimentResult:
    """
    训练单个超参数配置
    
    Parameters:
    -----------
    config : 超参数配置
    df_train : 训练数据
    df_test : 测试数据
    sentiment_dict : 情绪因子字典
    save_dir : 保存目录
    verbose : 是否打印详细信息
    
    Returns:
    --------
    ExperimentResult : 实验结果
    """
    import time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"开始训练配置: {config.name}")
        print(f"{'='*60}")
    
    # 设置随机种子
    if config.random_state is not None:
        set_seed(config.random_state)
    
    # 创建模型
    model = NeuralSDEPricer(
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        n_paths=config.n_paths,
        n_steps=config.n_steps,
        rho=config.rho,
        rho_source=config.rho_source,    # 使用配置中的rho来源
        residual_scale=config.residual_scale,
        loss_type=config.loss_type,      # 使用配置中的损失函数类型
        v0_source=config.v0_source,      # 使用配置中的初始方差来源
        random_state=config.random_state
    )
    
    # 训练
    start_time = time.time()
    model.fit(
        df_train=df_train,
        sentiment_dict=sentiment_dict,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        n_paths_train=config.n_paths_train,
        verbose=verbose
    )
    training_time = time.time() - start_time
    
    # 预测
    if verbose:
        print(f"\n开始预测测试集...")
    predictions, std_errors = model.predict_batch(
        df_test=df_test,
        sentiment_dict=sentiment_dict,
        n_paths_test=config.n_paths_test,
        verbose=verbose
    )
    
    # 计算指标
    targets = df_test['close'].values
    metrics = compute_metrics(targets, predictions)
    
    if verbose:
        print(f"\n配置 {config.name} 结果:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        if 'R2' in metrics:
            print(f"  R²: {metrics['R2']:.4f}")
        print(f"  训练时间: {training_time:.1f}秒")
    
    # 保存模型
    model_path = save_dir / f"model_{config.name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # 返回结果
    return ExperimentResult(
        config=config,
        metrics=metrics,
        predictions=predictions,
        targets=targets,
        training_history=model.history_,
        model_path=str(model_path),
        training_time=training_time
    )


def run_experiments(
    configs: List[HyperparameterConfig],
    sentiment_dict: Dict[str, np.ndarray],
    train_data_file: Optional[str] = None,
    test_data_file: Optional[str] = None,
    df_train: Optional[pd.DataFrame] = None,
    df_test: Optional[pd.DataFrame] = None,
    save_dir: str = 'hyperparameter_results',
    history_file: Optional[str] = None,  # 新增：自定义历史文件名
    verbose: bool = True,
    save_models: bool = True
) -> List[ExperimentResult]:
    """
    顺序运行多个超参数配置实验（支持缓存）
    
    注意：由于PyTorch模型无法在进程间序列化，使用顺序训练而非并行。
    这样更稳定，且能实时监控每个配置的训练进度。
    
    Parameters:
    -----------
    configs : 超参数配置列表
    sentiment_dict : 情绪因子字典
    train_data_file : 训练数据CSV文件路径（优先使用）
    test_data_file : 测试数据CSV文件路径（优先使用）
    df_train : 训练数据DataFrame（向后兼容，当未提供文件路径时使用）
    df_test : 测试数据DataFrame（向后兼容，当未提供文件路径时使用）
    save_dir : 结果保存目录
    history_file : 自定义历史文件名（默认为 'experiment_history.json'）
    verbose : 是否打印详细信息
    save_models : 是否保存训练好的模型（设为False可节省磁盘空间）
    
    Returns:
    --------
    results : 实验结果列表
    """
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # 创建models子目录
    models_dir = save_path / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # 数据加载：优先使用文件路径，否则使用DataFrame
    if train_data_file and test_data_file:
        if verbose:
            print(f"\n{'='*60}")
            print("数据加载:")
            print(f"  训练集: {train_data_file}")
            print(f"  测试集: {test_data_file}")
        df_train_used = pd.read_csv(train_data_file)
        df_test_used = pd.read_csv(test_data_file)
        if verbose:
            print(f"  训练集大小: {len(df_train_used)}")
            print(f"  测试集大小: {len(df_test_used)}")
    elif df_train is not None and df_test is not None:
        df_train_used = df_train
        df_test_used = df_test
        train_data_file = "DataFrame (in-memory)"
        test_data_file = "DataFrame (in-memory)"
        if verbose:
            print(f"\n使用内存中的DataFrame")
            print(f"  训练集大小: {len(df_train_used)}")
            print(f"  测试集大小: {len(df_test_used)}")
    else:
        raise ValueError("必须提供 train_data_file/test_data_file 或 df_train/df_test")
    
    # 加载实验历史
    if history_file:
        # 使用自定义历史文件名
        history_path = save_path / history_file
    else:
        # 使用默认历史文件名
        history_path = save_path / 'experiment_history.json'
    
    history = load_experiment_history(str(history_path))
    
    # 更新元数据
    history['metadata']['train_data_file'] = train_data_file
    history['metadata']['test_data_file'] = test_data_file
    history['metadata']['train_size'] = len(df_train_used)
    history['metadata']['test_size'] = len(df_test_used)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"加载实验历史... 已有 {len(history['experiments'])} 个实验记录")
        print(f"结果保存至: {save_path}")
        print(f"{'='*60}\n")
    
    results = []
    
    # 顺序训练每个配置
    for i, config in enumerate(configs, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{i}/{len(configs)}] 配置: {config.name}")
            print(f"{'='*60}")
        
        # 计算配置哈希
        config_hash = compute_config_hash(config)
        
        # 检查缓存
        cached_result = check_cached_result(
            config_hash, history, train_data_file, test_data_file
        )
        
        if cached_result is not None:
            # 使用缓存结果
            if verbose:
                cached_exp = history['experiments'][config_hash]
                print(f"  ✓ 找到缓存结果")
                print(f"    训练于: {cached_exp['trained_at']}")
                print(f"    MAE: {cached_exp['metrics']['MAE']:.4f}")
                print(f"    训练时间: {cached_exp['training_time']:.1f}秒")
                print(f"  跳过训练")
            
            results.append(cached_result)
            continue
        
        # 没有缓存，需要训练
        if verbose:
            print(f"  ✗ 未找到缓存")
            print(f"  开始训练...")
        
        try:
            # 训练单个配置
            result = train_single_config(
                config=config,
                df_train=df_train_used,
                df_test=df_test_used,
                sentiment_dict=sentiment_dict,
                save_dir=models_dir if save_models else save_path,  # 模型保存到models子目录
                verbose=verbose
            )
            
            results.append(result)
            
            # 保存模型到 models/ 子目录（如果需要）
            if save_models and result.model_path:
                old_model_path = Path(result.model_path)
                new_model_path = models_dir / f"{config_hash}.pkl"
                if old_model_path.exists():
                    import shutil
                    shutil.move(str(old_model_path), str(new_model_path))
                    result.model_path = str(new_model_path)
            
            # 立即更新历史记录
            history['experiments'][config_hash] = {
                'config': config.to_dict(),
                'metrics': result.metrics,
                'training_time': result.training_time,
                'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_path': f"models/{config_hash}.pkl" if save_models else None,
                'data_files': {
                    'train': train_data_file,
                    'test': test_data_file
                },
                'predictions': result.predictions.tolist() if len(result.predictions) < 10000 else [],
                'targets': result.targets.tolist() if len(result.targets) < 10000 else []
            }
            
            # 保存历史记录
            save_experiment_history(history, str(history_path))
            
            if verbose:
                print(f"\n✅ 配置 {config.name} 完成")
                print(f"   MAE: {result.metrics['MAE']:.4f}")
                print(f"   训练时间: {result.training_time:.1f}秒")
                
                # 显示进度
                completed = len(results)
                remaining = len(configs) - completed
                if completed > 0:
                    avg_time = sum(r.training_time for r in results) / completed
                    est_remaining_time = avg_time * remaining / 60
                    
                    print(f"\n📊 总体进度: {completed}/{len(configs)} 完成")
                    if remaining > 0:
                        print(f"   预计剩余时间: {est_remaining_time:.1f} 分钟")
            
            # 如果不保存模型，删除模型文件以节省空间
            if not save_models and result.model_path:
                model_path = Path(result.model_path)
                if model_path.exists():
                    model_path.unlink()
                    result.model_path = None
        
        except Exception as exc:
            print(f"\n❌ 配置 {config.name} 训练失败: {exc}")
            import traceback
            traceback.print_exc()
            continue
    
    # 最终保存历史记录
    save_experiment_history(history, str(history_path))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ 所有实验完成！")
        print(f"   成功: {len(results)}/{len(configs)}")
        print(f"   结果已保存至: {save_path / 'experiment_history.json'}")
        print(f"{'='*60}")
    
    return results


# 保留旧的函数名作为别名，避免破坏兼容性
def run_parallel_experiments(*args, **kwargs):
    """
    别名函数，实际调用 run_experiments
    
    注意：由于PyTorch模型序列化问题，实际使用顺序训练而非并行。
    n_workers 参数将被忽略。
    """
    # 移除 n_workers 参数（如果存在）
    kwargs.pop('n_workers', None)
    return run_experiments(*args, **kwargs)


def visualize_comparison(
    results: List[ExperimentResult],
    save_dir: str = 'hyperparameter_results',
    figsize: Tuple[int, int] = (32, 20)
):
    """
    可视化对比多个配置的结果
    
    Parameters:
    -----------
    results : 实验结果列表
    save_dir : 保存目录
    figsize : 图像大小
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # 准备数据
    config_names = [r.config.name for r in results]
    metrics_df = pd.DataFrame([r.metrics for r in results], index=config_names)
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 指标对比柱状图
    ax1 = fig.add_subplot(gs[0, :])
    metrics_to_plot = ['MAE', 'RMSE', 'MAPE']
    x = np.arange(len(config_names))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = metrics_df[metric].values
        ax1.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax1.set_xlabel('配置名称', fontsize=12)
    ax1.set_ylabel('指标值', fontsize=12)
    ax1.set_title('各配置性能指标对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(config_names, rotation=30, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 预测 vs 真实散点图（选取最优配置）
    best_idx = metrics_df['MAE'].argmin()
    best_result = results[best_idx]
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(best_result.targets, best_result.predictions, alpha=0.5, s=20)
    min_val = min(best_result.targets.min(), best_result.predictions.min())
    max_val = max(best_result.targets.max(), best_result.predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel('真实价格', fontsize=11)
    ax2.set_ylabel('预测价格', fontsize=11)
    ax2.set_title(f'最优配置: {best_result.config.name}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 残差分布（最优配置）
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = best_result.predictions - best_result.targets
    ax3.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('残差', fontsize=11)
    ax3.set_ylabel('频数', fontsize=11)
    ax3.set_title('残差分布（最优配置）', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 训练时间对比
    ax4 = fig.add_subplot(gs[1, 2])
    training_times = [r.training_time for r in results]
    colors = ['green' if i == best_idx else 'gray' for i in range(len(results))]
    ax4.barh(config_names, training_times, color=colors, alpha=0.7)
    ax4.set_xlabel('训练时间（秒）', fontsize=11)
    ax4.set_title('训练时间对比', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. 训练损失曲线（所有配置）
    ax5 = fig.add_subplot(gs[2, 0])
    for result in results:
        if result.training_history and 'loss' in result.training_history:
            ax5.plot(result.training_history['loss'], label=result.config.name, alpha=0.7)
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Loss', fontsize=11)
    ax5.set_title('训练损失曲线', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 6. MAE曲线
    ax6 = fig.add_subplot(gs[2, 1])
    for result in results:
        if result.training_history and 'mae' in result.training_history:
            ax6.plot(result.training_history['mae'], label=result.config.name, alpha=0.7)
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('MAE', fontsize=11)
    ax6.set_title('训练MAE曲线', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. 指标排名热力图
    ax7 = fig.add_subplot(gs[2, 2])
    rank_df = metrics_df[['MAE', 'RMSE', 'MAPE']].rank()   # 去掉了R2
    sns.heatmap(rank_df.T, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                cbar_kws={'label': '排名'}, ax=ax7)
    ax7.set_title('指标排名热力图（越小越好）', fontsize=12, fontweight='bold')
    ax7.set_xticklabels(config_names, rotation=45, ha='right')
    
    plt.suptitle('超参数调优对比分析', fontsize=16, fontweight='bold', y=0.995)
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(save_path / f'comparison_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"对比图表已保存至: {save_path / f'comparison_{timestamp}.png'}")
    
    plt.tight_layout()
    plt.show()
    
    # 保存指标表格
    metrics_df.to_csv(save_path / f'metrics_comparison_{timestamp}.csv')
    print(f"指标表格已保存至: {save_path / f'metrics_comparison_{timestamp}.csv'}")
    
    # 打印最优配置
    print(f"\n{'='*60}")
    print("最优配置:")
    print(f"  名称: {best_result.config.name}")
    print(f"  MAE: {best_result.metrics['MAE']:.4f}")
    print(f"  RMSE: {best_result.metrics['RMSE']:.4f}")
    print(f"  MAPE: {best_result.metrics['MAPE']:.2f}%")
    if 'R2' in best_result.metrics:
        print(f"  R²: {best_result.metrics['R2']:.4f}")
    print(f"  训练时间: {best_result.training_time:.1f}秒")
    print(f"{'='*60}")


def create_preset_configs() -> List[HyperparameterConfig]:
    """
    创建预设的超参数配置组
    
    Returns:
    --------
    configs : 配置列表
    """
    configs = [
        # 基准配置
        HyperparameterConfig(
            name='baseline',
            latent_dim=2,
            hidden_dims=[32, 32],
            n_paths=5000,
            n_steps=50,
            residual_scale=0.3,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            n_paths_train=1000,
            n_paths_test=10000,
            random_state=42
        ),
        
        # 更深的网络
        HyperparameterConfig(
            name='deeper_net',
            latent_dim=2,
            hidden_dims=[64, 64, 32],
            n_paths=5000,
            n_steps=50,
            residual_scale=0.3,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            n_paths_train=1000,
            n_paths_test=10000,
            random_state=42
        ),
        
        # 更大的残差缩放
        HyperparameterConfig(
            name='large_residual',
            latent_dim=2,
            hidden_dims=[32, 32],
            n_paths=5000,
            n_steps=50,
            residual_scale=0.5,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            n_paths_train=1000,
            n_paths_test=10000,
            random_state=42
        ),
        
        # 更小的残差缩放
        HyperparameterConfig(
            name='small_residual',
            latent_dim=2,
            hidden_dims=[32, 32],
            n_paths=5000,
            n_steps=50,
            residual_scale=0.1,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            n_paths_train=1000,
            n_paths_test=10000,
            random_state=42
        ),
        
        # 更高的学习率
        HyperparameterConfig(
            name='high_lr',
            latent_dim=2,
            hidden_dims=[32, 32],
            n_paths=5000,
            n_steps=50,
            residual_scale=0.3,
            epochs=50,
            batch_size=64,
            lr=5e-3,
            n_paths_train=1000,
            n_paths_test=10000,
            random_state=42
        ),
        
        # 更细的时间步长
        HyperparameterConfig(
            name='fine_steps',
            latent_dim=2,
            hidden_dims=[32, 32],
            n_paths=5000,
            n_steps=100,
            residual_scale=0.3,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            n_paths_train=1000,
            n_paths_test=10000,
            random_state=42
        ),
    ]
    
    return configs


def create_fast_test_configs() -> List[HyperparameterConfig]:
    """
    创建快速测试配置组（用于验证流程，训练速度快）
    
    这些配置使用较少的epochs和路径数，适合：
    - 首次运行验证流程
    - 快速对比不同超参数的趋势
    - 调试代码
    
    Returns:
    --------
    configs : 轻量级配置列表
    """
    configs = [
        HyperparameterConfig(
            name='fast_baseline',
            latent_dim=2,
            hidden_dims=[32, 32],
            n_paths=2000,
            n_steps=30,
            residual_scale=0.3,
            epochs=10,  # 只训练10轮
            batch_size=64,
            lr=1e-3,
            n_paths_train=500,  # 训练时只用500条路径
            n_paths_test=2000,  # 测试时只用2000条路径
            random_state=42
        ),
        
        HyperparameterConfig(
            name='fast_high_lr',
            latent_dim=2,
            hidden_dims=[32, 32],
            n_paths=2000,
            n_steps=30,
            residual_scale=0.3,
            epochs=10,
            batch_size=64,
            lr=5e-3,
            n_paths_train=500,
            n_paths_test=2000,
            random_state=42
        ),

        HyperparameterConfig(
            name='fast_high_lr_small_rs',
            latent_dim=2,
            hidden_dims=[32, 32],
            n_paths=2000,
            n_steps=30,
            residual_scale=0.2,
            epochs=10,
            batch_size=64,
            lr=5e-3,
            n_paths_train=500,
            n_paths_test=2000,
            random_state=42
        ),

        HyperparameterConfig(
            name='fast_high_lr_little_rs',
            latent_dim=2,
            hidden_dims=[32, 32],
            n_paths=2000,
            n_steps=30,
            residual_scale=0.1,
            epochs=10,
            batch_size=64,
            lr=5e-3,
            n_paths_train=500,
            n_paths_test=2000,
            random_state=42
        ),
    ]
    
    return configs


# =========================== 使用示例 ===========================

if __name__ == '__main__':
    # 示例：如何使用这个框架
    print("超参数调优框架已准备好！")
    print("\n使用方法:")
    print("1. 在 Notebook 中导入此模块")
    print("2. 准备数据: df_train, df_test, sentiment_dict")
    print("3. 创建配置列表（或使用预设配置）")
    print("4. 运行: results = run_experiments(configs, df_train, df_test, sentiment_dict)")
    print("5. 可视化: visualize_comparison(results)")
    print("\n预设配置:")
    print("  - create_preset_configs(): 6组完整配置（训练较慢但精确）")
    print("  - create_fast_test_configs(): 3组快速配置（用于验证流程）")
    print("\n配置示例:")
    configs = create_preset_configs()
    for cfg in configs:
        print(f"  - {cfg.name}")
    
    print("\n⚠️ 注意事项:")
    print("  - 由于PyTorch模型序列化限制，使用顺序训练而非并行")
    print("  - 建议先用 fast_test 配置验证流程")
    print("  - 每个配置的训练时间取决于数据量和硬件性能")
    print("  - 可以随时中断，已完成的结果会自动保存")
