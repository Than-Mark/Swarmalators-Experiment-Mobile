import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ma
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClusterMixin

class PeriodicDBSCAN(BaseEstimator, ClusterMixin):
    """考虑周期性边界条件的聚类算法"""
    def __init__(self, eps=0.5, min_samples=5, boundary=5):
        self.eps = eps
        self.min_samples = min_samples
        self.boundary = boundary  # 周期边界长度

    def _adjusted_distance(self, a, b):
        """计算周期性调整后的距离"""
        delta = np.abs(a - b)
        delta = np.where(delta > self.L/2, self.L - delta, delta)
        return np.linalg.norm(delta)
    
    def fit(self, X):
        self.labels_ = -np.ones(X.shape[0], dtype=int)
        cluster_id = 0
        
        for i in range(X.shape[0]):
            if self.labels_[i] != -1:
                continue
                
            # 寻找邻域粒子
            neighbors = []
            for j in range(X.shape[0]):
                if self._adjusted_distance(X[i], X[j]) <= self.eps:
                    neighbors.append(j)
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -2  # 标记为噪声
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
        return self
    
    def _expand_cluster(self, X, core_idx, neighbors, cluster_id):
        self.labels_[core_idx] = cluster_id
        queue = list(neighbors)
        
        while queue:
            j = queue.pop(0)
            if self.labels_[j] == -2:  # 噪声转为边界点
                self.labels_[j] = cluster_id
            if self.labels_[j] != -1:
                continue
                
            self.labels_[j] = cluster_id
            new_neighbors = []
            for k in range(X.shape[0]):
                if self._adjusted_distance(X[j], X[k]) <= self.eps:
                    new_neighbors.append(k)
            
            if len(new_neighbors) >= self.min_samples:
                queue.extend(new_neighbors)

def compute_cluster_order_parameters(model, d_th=0.3):
    """计算基于聚类的序参量"""
    # 读取粒子轨迹数据
    targetPath = f"./data/{model}.h5"
    totalPositionX = pd.read_hdf(targetPath, key="positionX")
    totalSpeed = pd.read_hdf(targetPath, key="speed")
    totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
    totalPointThetaSpeed = pd.read_hdf(targetPath, key="pointThetaSpeed")
    TNum = totalPositionX.shape[0] // model.agentsNum
    totalPositionX = totalPositionX.values.reshape(TNum, model.agentsNum, 2)
    totalSpeed = totalSpeed.values.reshape(TNum, model.agentsNum, 2)
    totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, model.agentsNum)
    totalPointThetaSpeed = totalPointThetaSpeed.values.reshape(TNum, model.agentsNum)
    position = totalPositionX[-1, :, :]
    phase = totalPhaseTheta[-1, :]
    omega = totalPointThetaSpeed[-1, :]
    
    # 计算旋转中心 
    X = position[:,0] - model.v * np.sin(phase) / omega
    Y = position[:,1] + model.v * np.cos(phase) / omega
    centers = np.column_stack([X, Y])
    
    # 执行周期性聚类
    clusterer = PeriodicDBSCAN(eps=d_th, min_samples=5, boundary=model.boundaryLength)
    cluster_labels = clusterer.fit_predict(centers)
    
   # 计算每个簇的特征（包含噪声点形成的单粒子簇）
    cluster_R = []
    cluster_Domega = []
    
    for label in np.unique(cluster_labels):
        mask = (cluster_labels == label)
        # 处理单粒子簇
        if mask.sum() == 1:
            cluster_R.append(1.0)  # 单个粒子的同步度为1
            cluster_Domega.append(0.0)  # 无频率差异
            continue
            
        # 类内同步度
        theta = phase[mask]
        R_k = np.abs(np.exp(1j*theta).mean())
        cluster_R.append(R_k)
        
        # 类内频率一致性
        omega_k = omega[mask]
        delta = np.mean((omega_k - omega_k.mean())**2)
        cluster_Domega.append(delta)
    
    # 全局平均值
    R_c = np.mean(cluster_R) if cluster_R else 0.0
    Domega = np.mean(cluster_Domega) if cluster_Domega else 0.0
    
    return R_c, Domega

def plot_2d_phase_diagram(models, x_param, y_param, fixed_params):

    # 参数校验
    validate_params = [x_param, y_param] + list(fixed_params.keys())
    for param in validate_params:
        if not hasattr(models[0], param):
            raise ValueError(f"Invalid parameter: {param}")

    # 筛选模型
    TOL = 1e-6
    filtered = []
    for model in models:
        match = all(
            np.isclose(getattr(model, key), val, atol=TOL)
            for key, val in fixed_params.items()
        )
        if match:
            filtered.append(model)

    if not filtered:
        print(f"No data for {fixed_params}")
        return

    # 提取动态数据
    data = {
        x_param: [],
        y_param: [],
        'R_c': [],
        'ΔΩ*': []
    }
    
    for model in filtered:
        R_c, Domega = compute_cluster_order_parameters(model)
        data[x_param].append(getattr(model, x_param))
        data[y_param].append(getattr(model, y_param))
        data['R_c'].append(R_c)
        data['ΔΩ*'].append(Domega)

    # 创建网格数据
    df = pd.DataFrame(data)
    try:
        pivot_R = df.pivot(index=y_param, columns=x_param, values='R_c')
        pivot_D = df.pivot(index=y_param, columns=x_param, values='ΔΩ*')
    except KeyError:
        print("Parameter combination not fully sampled")
        return

    # 生成固定参数标签
    fixed_label = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])

    # 可视化设置
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # R_c热图
    im1 = ax[0].imshow(pivot_R,
                      extent=[df[x_param].min(), df[x_param].max(),
                              df[y_param].min(), df[y_param].max()],
                      origin='lower', 
                      cmap='viridis',
                      aspect='auto')
    ax[0].set_title(f'R_c ({fixed_label})')
    ax[0].set_xlabel(x_param)
    ax[0].set_ylabel(y_param)
    plt.colorbar(im1, ax=ax[0], label='R_c')

    # ΔΩ热图
    im2 = ax[1].imshow(pivot_D,
                      extent=[df[x_param].min(), df[x_param].max(),
                              df[y_param].min(), df[y_param].max()],
                      origin='lower',
                      cmap='plasma',
                      aspect='auto')
    ax[1].set_title(f'ΔΩ ({fixed_label})')
    ax[1].set_xlabel(x_param)
    plt.colorbar(im2, ax=ax[1], label='ΔΩ')

    plt.tight_layout()
    filename = f"phase_{x_param}_{y_param}_" + "_".join([f"{k}{v}" for k,v in fixed_params.items()]) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    from itertools import product
    from main import PeriodicalPotential
    from multiprocessing import Pool

    rangeLambdas = np.concatenate([
        np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.2)
    ])
    distanceDs = np.concatenate([
        np.arange(0.1, 1, 0.2)
    ])
    rangeGamma = np.linspace(0, 2, 5)
    rangeKappa = np.linspace(0, 1, 5)
    rangePeriod = np.linspace(0.1, 2.5, 6)

    savePath = "./data"

    models = [
        PeriodicalPotential(
            strengthLambda=l, distanceD=d, gamma=g, kappa=k, L=period,
            agentsNum=1000, boundaryLength=5,
            tqdm=True, savePath=savePath, overWrite=False)
        for l, d, g, k, period in product(rangeLambdas, distanceDs, rangeGamma, rangeKappa, rangePeriod)
    ]
    
# 全参数组合生成
param_pairs = [
    ('strengthLambda', 'distanceD'),
    ('strengthLambda', 'gamma'),
    ('strengthLambda', 'kappa'),
    ('strengthLambda', 'L'),
    ('distanceD', 'kappa'),
    ('distanceD', 'gamma'),
    ('distanceD', 'L'),
    ('gamma', 'kappa'),
    ('gamma', 'L'),
    ('kappa', 'L'),
]

fixed_defaults = {
    'gamma': 1.0,
    'kappa': 0.5,
    'L': 1.25,
    'strengthLambda': 0.5,
    'distanceD': 0.5
}

for x_param, y_param in param_pairs:
    # 确定需要固定的参数
    fixed_params = {k:v for k,v in fixed_defaults.items() 
                   if k not in [x_param, y_param]}
    
    # 获取该组合的参数范围
    x_values = sorted({getattr(m, x_param) for m in models})
    y_values = sorted({getattr(m, y_param) for m in models})
    
    # 生成网格配置
    for x_val, y_val in product(x_values[:3], y_values[:3]):  # 示例采样
        current_fixed = fixed_params.copy()
        current_fixed.update({x_param: x_val, y_param: y_val})
        
        try:
            plot_2d_phase_diagram(models,
                                x_param=x_param,
                                y_param=y_param,
                                fixed_params=current_fixed)
        except Exception as e:
            print(f"Failed {x_param}-{y_param} at {current_fixed}: {str(e)}")

    with Pool(40) as p:
        p.map(plot_2d_phase_diagram(models, x_param, y_param, fixed_params), models)