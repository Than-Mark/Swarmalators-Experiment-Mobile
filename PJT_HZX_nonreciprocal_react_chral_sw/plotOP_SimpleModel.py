import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

from main import StateAnalysis, ChiralActiveMatterNonreciprocalReact

CMAP = LinearSegmentedColormap.from_list('custom_blue', ['#f7fbff', '#08306b'])

# 数据文件夹路径
data_folder = r"data"
analysis_results_folder = r"D:\xys\Documents\paper\Swarmalator\Swarmalators-Experiment\PJT_HZX_spatial_noise_chiral_swarmalator\analysis_results"
os.makedirs(analysis_results_folder, exist_ok=True)

# 定义参数范围
param_sets = [
    {
        'strengthLambda': [0.04],
        'distanceD0Mean1': [0.1, 0.2, 0.35, 0.8],  # x轴参数
        'distanceD0Mean2': [0.1, 0.2, 0.35, 0.8]   # y轴参数
    },
]

omegaDistribution = "uniform"
chiralNum = 2
randomSeed = 10

# 生成模型实例
models = [
    ChiralActiveMatterNonreciprocalReact(
        strengthLambda=strengthLambda,
        distanceD0Mean1=d0m1,
        distanceD0Mean2=d0m2,
        chiralNum=chiralNum,
        agentsNum=1000,
        dt=0.01,
        tqdm=True,
        savePath=data_folder,
        shotsnaps=5,
        omegaDistribution=omegaDistribution,
        randomSeed=randomSeed,
        overWrite=True
    )
    for param_set in param_sets
    for strengthLambda in param_set['strengthLambda']
    for d0m1 in param_set['distanceD0Mean1']
    for d0m2 in param_set['distanceD0Mean2']
]

def process_model(model):
    sa = StateAnalysis(model)
    lookIdxs = np.arange(-2, 0)
    half_num = model.agentsNum // 2
    
    # 公共计算函数
    def calc_metrics(phases, positions, pointThetas):  # 修改函数签名
    # 频率差异修正
        Delta_Omega = np.std([
        p / pt 
        for p, pt in zip(phases, pointThetas)  # 添加pointThetas参数
        ], axis=0).mean()

        # 相位同步
        R = [StateAnalysis._clac_phase_sync_op(p) for p in phases]
        
        # 局部聚类
        Rc = []
        for idx in lookIdxs:
            centers = np.mod(sa.centers, 10)
            classes = StateAnalysis._calc_classes(centers, 0.3, 
                StateAnalysis._adj_distance(centers, centers[:, np.newaxis], 10, 5))
            valid_classes = [c for c in classes if len(c) > 5]
            if not valid_classes: 
                Rc.append(0)
                continue
            Rc.append(np.mean([StateAnalysis._clac_phase_sync_op(phases[idx][c]) for c in valid_classes]))
        
        # 空间凝聚力
        Nr = np.mean([len(max(classes, key=len))/model.agentsNum for classes in valid_classes])
        
        return {
            'R': np.mean(R),
            'Rc': np.mean(Rc),
            'Delta_Omega': Delta_Omega,
            'Nr': cluster_density
        }

    # 在调用时传递正确的pointThetas
    global_result = calc_metrics(
        [sa.totalPhaseTheta[idx] for idx in lookIdxs],
        [sa.totalPositionX[idx] for idx in lookIdxs],
        [sa.totalPointTheta[idx] for idx in lookIdxs]  # 新增参数
    )
    
    # 手性1计算
    chiral1_result = calc_metrics(
        [sa.totalPhaseTheta[idx][:half_num] for idx in lookIdxs],
        [sa.totalPositionX[idx][:half_num] for idx in lookIdxs]
    )
    
    # 手性2计算 
    chiral2_result = calc_metrics(
        [sa.totalPhaseTheta[idx][half_num:] for idx in lookIdxs],
        [sa.totalPositionX[idx][half_num:] for idx in lookIdxs]
    )

    return {
        'd0m1': model.distanceD0Mean1,
        'd0m2': model.distanceD0Mean2,
        **{f'Global_{k}': v for k,v in global_result.items()},
        **{f'Chiral1_{k}': v for k,v in chiral1_result.items()},
        **{f'Chiral2_{k}': v for k,v in chiral2_result.items()}
    }


def create_unified_heatmap(data, d0m1_values, d0m2_values, metrics, prefix):
    # 生成每个指标的热图
    for metric in metrics:
        # 获取当前metric数据
        metric_data = [item[f'{prefix}_{metric}'] for item in data]
        
        # 确定颜色范围
        vmin, vmax = np.nanmin(metric_data), np.nanmax(metric_data)
        
        # 创建网格
        grid = np.full((len(d0m2_values), len(d0m1_values)), np.nan)
        for item in data:
            i = d0m2_values.index(item['d0m2'])
            j = d0m1_values.index(item['d0m1'])
            grid[i, j] = item[f'{prefix}_{metric}']
        
        # 可视化
        plt.figure(figsize=(8,6))
        plt.imshow(grid, cmap=CMAP, origin='lower', 
                  extent=[min(d0m1_values), max(d0m1_values), 
                          min(d0m2_values), max(d0m2_values)],
                  vmin=vmin, vmax=vmax)
        
        plt.colorbar(label=metric)
        plt.xlabel('distanceD0Mean1')
        plt.ylabel('distanceD0Mean2')
        plt.title(f'{prefix} - {metric}')
        
        # 添加数值标签
        for i in range(len(d0m2_values)):
            for j in range(len(d0m1_values)):
                plt.text(d0m1_values[j], d0m2_values[i], f"{grid[i,j]:.2f}",
                         ha='center', va='center', color='w' if grid[i,j] > (vmax-vmin)/2 else 'k')
        
        plt.savefig(os.path.join(analysis_results_folder, f'Heatmap_L0.04_{prefix}_{metric}.png'))
        plt.close()


if __name__ == "__main__":
    # 多进程计算
    with Pool(12) as pool:
        results = list(tqdm(pool.imap(process_model, models), total=len(models)))
    
    # 获取参数范围
    d0m1_values = sorted(list(set(item['d0m1'] for item in results)))
    d0m2_values = sorted(list(set(item['d0m2'] for item in results)))
    
    # 需要可视化的指标列表
    metrics = ['R', 'Rc', 'Delta_Omega', 'Nr']
    
    # 生成三种类型的热图
    create_unified_heatmap(results, d0m1_values, d0m2_values, metrics, 'Global')
    create_unified_heatmap(results, d0m1_values, d0m2_values, metrics, 'Chiral1')
    create_unified_heatmap(results, d0m1_values, d0m2_values, metrics, 'Chiral2')