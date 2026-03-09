"""
VolatilitySurfaceVisualizer 使用示例
演示如何使用波动率曲面可视化工具进行分析

使用场景：
1. 快速查看某一天的波动率结构
2. 对比多个日期的市场情绪变化
3. 批量生成报告图表
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # project root

from VolatilitySurfaceVisualizer import VolatilitySurfaceVisualizer
import matplotlib.pyplot as plt

# ============================================================
# 示例 1: 基础用法 - 分析单个日期
# ============================================================

def example_1_basic():
    """最简单的用法：一键生成所有图表"""
    print("\n" + "="*60)
    print("示例 1: 基础用法")
    print("="*60)
    
    # 创建可视化器
    viz = VolatilitySurfaceVisualizer(csv_path="full_option_trading_data.csv")
    
    # 加载数据
    viz.load_data(date="2024-02-05")
    
    # 一键生成所有图表
    fig_vol, fig_rr = viz.plot_all(
        save_path="outputs/example1/2024-02-05",
        show=True
    )
    
    print("✓ 示例 1 完成")


# ============================================================
# 示例 2: 高级用法 - 分步调用
# ============================================================

def example_2_advanced():
    """分步调用各个方法，更灵活地控制参数"""
    print("\n" + "="*60)
    print("示例 2: 高级用法 - 分步调用")
    print("="*60)
    
    viz = VolatilitySurfaceVisualizer(csv_path="full_option_trading_data.csv")
    
    # 加载数据时启用虚值程度过滤
    viz.load_data(
        date="2024-02-05",
        moneyness_filter=True  # 只保留虚值期权
    )
    
    # 打印诊断信息
    viz.print_diagnostics()
    
    # 单独绘制波动率曲面，使用更高分辨率
    fig_vol = viz.plot_volatility_surfaces(
        grid_resolution=80,  # 提高插值分辨率
        figsize=(20, 14),    # 更大的图形
        elev=25,             # 调整视角
        azim=50
    )
    
    # 单独绘制风险逆转曲面
    fig_rr = viz.plot_risk_reversal_surface(
        dist_max=0.20,       # 扩展到 20% 的虚值深度
        grid_resolution=60,
        figsize=(12, 8),
        elev=30,
        azim=-45
    )
    
    # 手动保存
    fig_vol.savefig("outputs/example2/volatility_high_res.png", dpi=300)
    fig_rr.savefig("outputs/example2/risk_reversal_wide.png", dpi=300)
    
    plt.show()
    
    print("✓ 示例 2 完成")


# ============================================================
# 示例 3: 批量分析 - 对比多个日期
# ============================================================

def example_3_batch():
    """批量分析多个日期，对比市场情绪变化"""
    print("\n" + "="*60)
    print("示例 3: 批量分析多个日期")
    print("="*60)
    
    # 选择几个关键日期
    dates = [
        "2024-02-01",  # 春节前
        "2024-02-05",  # 恐慌日
        "2024-02-19",  # 恢复期
    ]
    
    viz = VolatilitySurfaceVisualizer(csv_path="full_option_trading_data.csv")
    
    for date in dates:
        print(f"\n--- 正在分析 {date} ---")
        
        try:
            viz.load_data(date=date)
            
            # 只生成风险逆转图（情绪指标）
            fig_rr = viz.plot_risk_reversal_surface()
            fig_rr.savefig(f"outputs/example3/risk_reversal_{date}.png", dpi=150)
            
            plt.close(fig_rr)  # 关闭图形释放内存
            
            print(f"✓ {date} 分析完成")
            
        except Exception as e:
            print(f"✗ {date} 分析失败: {e}")
    
    print("\n✓ 批量分析完成，所有图片已保存到 outputs/example3/")


# ============================================================
# 示例 4: 自定义分析 - 提取数据进行二次处理
# ============================================================

def example_4_custom():
    """获取数据后进行自定义分析"""
    print("\n" + "="*60)
    print("示例 4: 自定义分析")
    print("="*60)
    
    viz = VolatilitySurfaceVisualizer(csv_path="full_option_trading_data.csv")
    viz.load_data(date="2024-02-05")
    
    # 访问内部数据进行自定义分析
    call_df = viz.call_df
    put_df = viz.put_df
    
    # 例如：计算 ATM 附近的 IV 差异
    call_atm = call_df[
        (call_df['log_moneyness(ln(K/S))'] >= -0.02) & 
        (call_df['log_moneyness(ln(K/S))'] <= 0.02)
    ]
    
    put_atm = put_df[
        (put_df['log_moneyness(ln(K/S))'] >= -0.02) & 
        (put_df['log_moneyness(ln(K/S))'] <= 0.02)
    ]
    
    print(f"\nATM 附近的 IV 分析:")
    print(f"  Call ATM IV 均值: {call_atm['iv'].mean():.4f}")
    print(f"  Put ATM IV 均值: {put_atm['iv'].mean():.4f}")
    print(f"  Put-Call Spread: {put_atm['iv'].mean() - call_atm['iv'].mean():.4f}")
    
    # 继续使用内置方法绘图
    viz.plot_all(save_path="outputs/example4/2024-02-05", show=False)
    
    print("✓ 示例 4 完成")


# ============================================================
# 示例 5: 错误处理
# ============================================================

def example_5_error_handling():
    """演示如何处理常见错误"""
    print("\n" + "="*60)
    print("示例 5: 错误处理")
    print("="*60)
    
    viz = VolatilitySurfaceVisualizer(csv_path="full_option_trading_data.csv")
    
    # 错误 1: 日期不存在
    try:
        viz.load_data(date="2025-12-31")
    except ValueError as e:
        print(f"✗ 预期错误: {e}")
    
    # 错误 2: 文件不存在
    try:
        viz_bad = VolatilitySurfaceVisualizer(csv_path="nonexistent.csv")
        viz_bad.load_data(date="2024-02-05")
    except FileNotFoundError as e:
        print(f"✗ 预期错误: 文件未找到")
    
    # 错误 3: 在加载数据前绘图
    try:
        viz_empty = VolatilitySurfaceVisualizer(csv_path="full_option_trading_data.csv")
        viz_empty.plot_volatility_surfaces()
    except ValueError as e:
        print(f"✗ 预期错误: {e}")
    
    print("\n✓ 错误处理示例完成")


# ============================================================
# 主函数：运行所有示例
# ============================================================

if __name__ == "__main__":
    import os
    
    # 创建输出目录
    os.makedirs("outputs/example1", exist_ok=True)
    os.makedirs("outputs/example2", exist_ok=True)
    os.makedirs("outputs/example3", exist_ok=True)
    os.makedirs("outputs/example4", exist_ok=True)
    
    print("\n" + "="*60)
    print("VolatilitySurfaceVisualizer 使用示例集")
    print("="*60)
    
    # 运行示例（可以注释掉不需要的）
    example_1_basic()           # 基础用法
    # example_2_advanced()        # 高级用法
    # example_3_batch()           # 批量分析
    # example_4_custom()          # 自定义分析
    # example_5_error_handling()  # 错误处理
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
