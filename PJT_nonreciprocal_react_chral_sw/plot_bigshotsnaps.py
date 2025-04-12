import matplotlib.pyplot as plt
import numpy as np
from main import ChiralActiveMatterNonreciprocalReact, StateAnalysis
import os
import matplotlib.colors as mcolors
from io import BytesIO
from itertools import product

# ==================== 配置部分 ====================
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# 参数配置
param_matrix = {
    "strengthLambda": [0.04],
    "distanceD0Mean1": [0.1, 0.2, 0.3, 0.8],
    "distanceD0Mean2": [0.1, 0.2, 0.3, 0.8],
    "chiralNum": [2],
    "omegaDistribution": ["uniform"],
    "randomSeed": [10]
}

# 时间步配置 (10000, 20000, ..., 60000)
time_steps = np.arange(10000, 60001, 10000)  # 共6个时间点
n_time_steps = len(time_steps)  # 6个子图列

# ==================== 核心函数 ====================
class FixedChiralModel(ChiralActiveMatterNonreciprocalReact):
    def __str__(self) -> str:
        """确保文件名包含完整参数信息"""
        return (
            f"ChiralActiveMatterNonreciprocalReact_"
            f"{self.chiralNum}_"
            f"{self.omegaDistribution}_"
            f"{self.strengthLambda:.2f}_"
            f"d0_{self.distanceD0Mean1}_{self.distanceD0Mean2}_"
            f"{self.randomSeed}"
        )

def plot_single_frame(params, t):
    """绘制单个帧并返回图像数据"""
    try:
        # 初始化模型
        model = FixedChiralModel(
            chiralNum=params["chiralNum"],
            strengthLambda=params["strengthLambda"],
            distanceD0Mean1=params["distanceD0Mean1"],
            distanceD0Mean2=params["distanceD0Mean2"],
            omegaDistribution=params["omegaDistribution"],
            savePath="./data/",
            randomSeed=params["randomSeed"],
            overWrite=False
        )
        
        # 调试输出
        print(f"\nLoading: {model}.h5")
        
        # 加载数据
        sa = StateAnalysis(model)
        total_steps = len(sa.totalPositionX)

        # 确保时间步在有效范围内
        if t < 2000 or t > 12000:
            raise ValueError(f"Time step {t} out of valid range (10000-60000)")
        
        # 计算索引 (t=10000 -> idx=0, t=20000 -> idx=1, ..., t=60000 -> idx=5)
        idx = (t // 10000) - 1
        if idx < 0 or idx >= total_steps:
            raise ValueError(f"Invalid index {idx} for t={t} (max step={total_steps*10000})")
        
        print(f"Processing: d1={params['distanceD0Mean1']} d2={params['distanceD0Mean2']} t={t} (idx={idx})")

        # 获取数据
        position = sa.totalPositionX[idx]
        phase = sa.totalPhaseTheta[idx]
        omega = model.omegaTheta  # 用于颜色映射

        # 创建画布
        fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # 颜色映射配置
        alpha_rate = 0.8
        half = model.agentsNum // 2
        
        # 绘制正手性粒子 (红色，omega在[1,3])
        alphas_red = (np.abs(omega[:half]) - 1) / 2 * alpha_rate + (1 - alpha_rate)
        for i in range(half):
            ax.quiver(
                position[i, 0], position[i, 1],
                np.cos(phase[i]), np.sin(phase[i]),
                color=(1, 0, 0, alphas_red[i]),  # RGBA格式
                scale=25,
                width=0.005
            )
        
        # 绘制负手性粒子 (蓝色，omega在[-3,-1])
        alphas_blue = (np.abs(omega[half:]) - 1) / 2 * alpha_rate + (1 - alpha_rate)
        for i in range(half):
            ax.quiver(
                position[half+i, 0], position[half+i, 1],
                np.cos(phase[half+i]), np.sin(phase[half+i]),
                color=(0, 0.25, 0.78, alphas_blue[i]),  # #414CC7的RGB值
                scale=25,
                width=0.005
            )
        
        ax.set(xlim=(0,10), ylim=(0,10))
        ax.axis('off')
        
        # 保存到缓冲区
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        return params, t, buf.getvalue()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return params, t, None

# ==================== 图片生成逻辑 ====================
def organize_data(results):
    """重组数据为三维字典结构"""
    data_dict = {}
    for params, t, img_data in results:
        if img_data:
            d1 = params["distanceD0Mean1"]
            d2 = params["distanceD0Mean2"]
            if d1 not in data_dict:
                data_dict[d1] = {}
            if d2 not in data_dict[d1]:
                data_dict[d1][d2] = {}
            data_dict[d1][d2][t] = img_data
    return data_dict

def create_figure(data_dict, d1):
    """创建单张大图"""
    fig, axs = plt.subplots(4, 6, figsize=(24, 16))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # 全局标题
    fig.suptitle(
        f"λ=0.04 | d1={d1}\n",
        y=0.95,
        fontsize=18
    )
    
    # 添加列标签（时间步）
    for col_idx, t in enumerate(time_steps):
        axs[0, col_idx].text(
            0.5, 1.2, f"{t//1000}k",
            transform=axs[0, col_idx].transAxes,
            ha='center',
            fontsize=12
        )
    
    # 添加行标签（d2值）
    d2_values = [0.1, 0.2, 0.3, 0.8]
    for row_idx, d2 in enumerate(d2_values):
        axs[row_idx, 0].text(
            -0.3, 0.5, f"d2={d2}",
            transform=axs[row_idx, 0].transAxes,
            rotation=90,
            va='center',
            fontsize=12
        )
    
    # 填充子图内容
    for row_idx, d2 in enumerate(d2_values):
        for col_idx, t in enumerate(time_steps):
            ax = axs[row_idx, col_idx]
            ax.axis('off')
            
            if d1 in data_dict and d2 in data_dict[d1] and t in data_dict[d1][d2]:
                img = plt.imread(BytesIO(data_dict[d1][d2][t]))
                ax.imshow(img)
    
    return fig

# ==================== 主流程 ====================
def main():
    # 生成所有参数组合
    all_params = [dict(zip(param_matrix.keys(), values)) 
                for values in product(*param_matrix.values())]
    
    # 顺序处理任务
    results = []
    for params in all_params:
        for t in time_steps:
            results.append(plot_single_frame(params, t))
    
    # 重组数据
    data_dict = organize_data(results)
    
    # 生成图片
    for d1 in [0.1, 0.2, 0.3, 0.8]:
        if d1 in data_dict:
            fig = create_figure(data_dict, d1)
            output_path = f"./png/lambda_0.04_d1_{d1}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Generated: {output_path}")

if __name__ == "__main__":
    main()