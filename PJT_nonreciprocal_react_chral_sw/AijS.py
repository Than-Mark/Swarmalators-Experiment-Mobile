# File: plot_bigshotsnaps_final_adjusted.py
import matplotlib.pyplot as plt
import numpy as np
from main import ChiralActiveMatterNonreciprocalReact, StateAnalysis
import multiprocessing
from tqdm import tqdm
import pandas as pd
import os
import sys

# ==================== 配置部分 ====================
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

param_matrix = {
    "strengthLambda": [0.04],
    "distanceD0Mean1": [0.1, 0.2, 0.35, 0.8, 1.75],
    "distanceD0Mean2": [0.1, 0.2, 0.35, 0.8, 1.75],
    "omegaDistribution": ["uniform"],
    "agentsNum": [2000],
    "randomSeed": [10]
}

NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 12)

class AdjustedChiralModel(ChiralActiveMatterNonreciprocalReact):
    def __str__(self) -> str:
        d0_2_for_str = self.distanceD0Mean2 if self.chiralNum == 2 else 0
        return (
            f"ChiralActiveMatterNonreciprocalReact_"
            f"{self.chiralNum}_{self.agentsNum}_uniform_"
            f"0.04_"
            f"d0_{self.distanceD0Mean1}_{d0_2_for_str}_"
            f"{self.randomSeed}"
        )

def calculate_rho_single(position_data, d_value, agents_per_time):
    """计算单手性 rho 值（向量化优化版）"""
    TNum = position_data.shape[0] // agents_per_time
    rho_avg_over_time = []
    
    for t in range(0, TNum, 5):
        # 提取当前时间步的位置数据
        pos = position_data[t*agents_per_time : (t+1)*agents_per_time]
        
        # 向量化计算距离矩阵
        dx = pos[:, 0, np.newaxis] - pos[:, 0]
        dy = pos[:, 1, np.newaxis] - pos[:, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # 计算邻接矩阵
        A = (distances <= d_value).astype(int)
        Ai = A.sum(axis=1)
        
        # 计算 rho
        area_circle = np.pi * (d_value**2)
        area_remaining = 100 - area_circle
        rho_i = (Ai / 10 - area_circle) / area_remaining
        rho_avg = np.mean(rho_i)
        
        rho_avg_over_time.append(rho_avg)
    
    return np.array(rho_avg_over_time)

def calculate_rho_double(position_data, d1, d2, total_agents):
    """计算双手性 rho 值（向量化优化版）"""
    TNum = position_data.shape[0] // total_agents
    half = total_agents // 2
    rho_d1 = []
    rho_d2 = []
    
    for t in range(TNum):
        pos = position_data[t*total_agents : (t+1)*total_agents]
        
        # 分割红蓝粒子
        pos_red = pos[:half]
        pos_blue = pos[half:]
        
        # 计算红粒子的邻接矩阵
        dx_red = pos_red[:, 0, np.newaxis] - pos[:, 0]
        dy_red = pos_red[:, 1, np.newaxis] - pos[:, 1]
        distances_red = np.sqrt(dx_red**2 + dy_red**2)
        A_red = (distances_red <= d1).astype(int)
        Ai_red = A_red.sum(axis=1)
        
        # 计算蓝粒子的邻接矩阵
        dx_blue = pos_blue[:, 0, np.newaxis] - pos[:, 0]
        dy_blue = pos_blue[:, 1, np.newaxis] - pos[:, 1]
        distances_blue = np.sqrt(dx_blue**2 + dy_blue**2)
        A_blue = (distances_blue <= d2).astype(int)
        Ai_blue = A_blue.sum(axis=1)
        
        # 计算 rho
        area_red = np.pi * (d1**2)
        area_blue = np.pi * (d2**2)
        area_remaining_red = 100 - np.array(area_red)
        area_remaining_blue = 100 - np.array(area_blue)
        
        rho_red = (Ai_red / 10 - area_red) / area_remaining_red
        rho_blue = (Ai_blue / 10 - area_blue) / area_remaining_blue
        
        rho_d1.append(np.mean(rho_red))
        rho_d2.append(np.mean(rho_blue))
    
    return np.array(rho_d1), np.array(rho_d2)

def plot_single_d1_figure(d1_value, base_params):
    print(f"开始绘制 d1 = {d1_value} 的图像...")
    # ----------------- 初始化图形和布局 -----------------
    fig = plt.figure(figsize=(44, 24), dpi=150)
    gs = fig.add_gridspec(nrows=6, ncols=11, wspace=0.05, hspace=0.05)
    plt.subplots_adjust(left=0.05)

    # 参数提取
    strength_lambda = base_params["strengthLambda"][0]
    omega_dist = "uniform"
    random_seed = 10
    d2_values = base_params["distanceD0Mean2"]
    actual_agents_num = base_params["agentsNum"][0]
    
    total_rows = 1 + len(d2_values)
    total_cols = 7

# ==================== 生成行标签 ====================
    row_titles = [f"d1={d1_value}"] + \
                [f"d1={d1_value}\nd2={d2}" for d2 in d2_values]


    # ----------------- 预先创建所有子图 -----------------
    ax_quiver = {}
    ax_plot = {}
    
    for i in range(total_rows):
        for j in range(total_cols):
            ax_quiver[(i,j)] = fig.add_subplot(gs[i, j])
            ax_quiver[(i,j)].axis('off')
            if j == 0:
                ax_quiver[(i,j)].text(
                    x=-0.05,  # 水平位置调整
                    y=0.5,
                    s=row_titles[i],
                    rotation=90,
                    va='center',
                    ha='right',
                    fontsize=8,
                    transform=ax_quiver[(i,j)].transAxes
                )
        ax_plot[i] = fig.add_subplot(gs[i, 7:9])
  
    # ==================== 数据加载 ====================
    # 单手性数据
    model_single = AdjustedChiralModel(
        chiralNum=1,
        agentsNum=actual_agents_num,
        strengthLambda=strength_lambda,
        distanceD0Mean1=d1_value,
        distanceD0Mean2=0,
        omegaDistribution=omega_dist,
        savePath="./data/",
        randomSeed=random_seed,
        overWrite=False
    )
    target_path_single = f"{model_single.savePath}/{model_single}.h5"
    positionX_single = pd.read_hdf(target_path_single, key="positionX").values.reshape(-1, 2)
    phase_data_single = pd.read_hdf(target_path_single, key="phaseTheta").values.reshape(-1, 1)
    single_agents = actual_agents_num // 2
    
    # 计算单手性 rho
    rho_single = calculate_rho_single(
        positionX_single, d1_value, single_agents
    )

    # 双手性数据
    rho_d1_dict = {}
    rho_d2_dict = {}
    positionX_double = {}
    phase_data_double = {}
    for d2 in d2_values:
        model = AdjustedChiralModel(
            chiralNum=2,
            agentsNum=actual_agents_num,
            strengthLambda=strength_lambda,
            distanceD0Mean1=d1_value,
            distanceD0Mean2=d2,
            omegaDistribution=omega_dist,
            savePath="./data/",
            randomSeed=random_seed,
            overWrite=False
        )
        target_path = f"{model.savePath}/{model}.h5"
        pos_data = pd.read_hdf(target_path, key="positionX").values.reshape(-1, 2)
        phaseTheta_data = pd.read_hdf(target_path, key="phaseTheta").values.reshape(-1, 1)
        
        positionX_double[d2] = pos_data
        phase_data_double[d2] = phaseTheta_data
        
        # 计算双手性 rho
        rho_d1, rho_d2 = calculate_rho_double(
            pos_data, d1_value, d2, actual_agents_num
        )
        rho_d1_dict[d2] = rho_d1
        rho_d2_dict[d2] = rho_d2

    # ==================== 绘图逻辑 ====================
    # 绘制单手性行
    alpha_rate = 0.8
    time_indices = np.linspace(0, len(rho_single)-1, total_cols, dtype=int)
    for j in range(total_cols):
        ax = ax_quiver[(0,j)]
        t_idx = time_indices[j]
        
        # 获取位置数据
        pos = positionX_single[t_idx*single_agents : (t_idx+1)*single_agents]
        phaseTheta = phase_data_single[t_idx*single_agents : (t_idx+1)*single_agents]
        omega = model_single.omegaTheta  
        alphas = (np.abs(omega) - 1) / 2 * alpha_rate + (1 - alpha_rate)

        # 绘制箭头
        single_colors = np.zeros((single_agents, 4))          # 形状 (1000, 4)
        single_colors[:, :3] = [1, 0, 0]             # RGB 设为红色
        single_colors[:, 3] = alphas[:single_agents]          # 透明度

        ax.quiver(
                pos[:,0], pos[:,1],        
                np.cos(phaseTheta), np.sin(phaseTheta),  
                color=single_colors,                     
                scale=25, width=0.005, headlength=5, headwidth=3, headaxislength=4
            )
        col_titles = ["t=0", "t=10k", "t=20k", "t=30k", "t=40k", "t=50k", "t=60k"]
        ax.set_title(col_titles[j], fontsize=8, pad=4)    #行标签
        ax.set(xlim=(0,10), ylim=(0,10))
        # 绘制折线图
        if j == 0:
            ax_plot[0].plot(rho_single, color='black', label='Single')
            ax_plot[0].set(xlabel='t', ylabel='rho')

    # 绘制双手性行
    for row_idx, d2 in enumerate(d2_values, start=1):
        pos_data = positionX_double[d2]
        phase_data = phase_data_double[d2]
        TNum = len(rho_d1_dict[d2])
        time_indices = np.linspace(0, TNum-1, total_cols, dtype=int)
        
        for j in range(total_cols):
            ax = ax_quiver[(row_idx,j)]
            t_idx = time_indices[j]
            
            # 获取位置数据
            pos = pos_data[t_idx*actual_agents_num : (t_idx+1)*actual_agents_num]
            phase = phase_data[t_idx*actual_agents_num : (t_idx+1)*actual_agents_num]
            omega = model.omegaTheta
            alphas = (np.abs(omega) - 1) / 2 * alpha_rate + (1 - alpha_rate)
            half = actual_agents_num // 2
            
            # 绘制双色箭头
            red_colors = np.zeros((half, 4))          # 形状 (1000, 4)
            red_colors[:, :3] = [1, 0, 0]             # RGB 设为红色
            red_colors[:, 3] = alphas[:half]          # 透明度

            blue_colors = np.zeros((half, 4))         # 形状 (1000, 4)
            blue_colors[:, :3] = [0, 0.25, 0.78]      # RGB 设为蓝色
            blue_colors[:, 3] = alphas[half:]         # 透明度

            ax.quiver(
                pos[:half, 0], pos[:half, 1],        # 所有红粒子的位置
                np.cos(phase[:half]), np.sin(phase[:half]),  # 所有红粒子的方向
                color=red_colors,                     # 正确颜色数组
                scale=25, width=0.005,
                headlength=5, headwidth=3, headaxislength=4
            )
            ax.quiver(
                pos[half:, 0], pos[half:, 1],        # 所有蓝粒子的位置
                np.cos(phase[half:]), np.sin(phase[half:]),  # 所有蓝粒子的方向
                color=blue_colors,                    # 正确颜色数组
                scale=25, width=0.005,
                headlength=5, headwidth=3, headaxislength=4
            )
            ax.set(xlim=(0,10), ylim=(0,10))
            
            # 绘制折线图
            if j == 0:
                ax_plot[row_idx].plot(
                    rho_d1_dict[d2], color='red', label=f'd1={d1_value}'
                )
                ax_plot[row_idx].plot(
                    rho_d2_dict[d2], color='blue', label=f'd2={d2}'
                )
                ax_plot[row_idx].set(xlabel='t', ylabel='rho')

    # ==================== 保存图像 ====================
    plt.tight_layout()
    output_filename = f"bigsnapshot_d1_{d1_value}.png"
    plt.savefig(os.path.join('./png/', output_filename), bbox_inches='tight')
    print(f"图像已保存到: {output_filename}")

    plt.close(fig)

# --- 定义包装函数---
def plot_single_d1_wrapper(args):
    return plot_single_d1_figure(*args)

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    multiprocessing.freeze_support() 

    # 提取基础参数 (除了 d1)
    base_params = {key: val for key, val in param_matrix.items() if key != "distanceD0Mean1"}
    d1_values_to_plot = param_matrix["distanceD0Mean1"]

    print("开始生成快照图像...")

    args_list = [(d1, base_params) for d1 in d1_values_to_plot]
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        list(tqdm(pool.imap_unordered(plot_single_d1_wrapper, args_list),
            total=len(args_list), desc="Processing d1 values (parallel)"))

    print("所有图像生成完毕!")