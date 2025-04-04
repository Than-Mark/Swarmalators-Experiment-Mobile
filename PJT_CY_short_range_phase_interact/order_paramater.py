import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm.notebook import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import sys
import os
import shutil

randomSeed = 100


new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
)

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

import seaborn as sns
import matplotlib.font_manager as fm

sns.set_theme(font_scale=1.1, rc={
    'figure.figsize': (6, 5),
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.5,
    "lines.linewidth": 1.5,
    'text.color': '#000000',
    'figure.titleweight': "bold",
    'xtick.color': '#000000',
    'ytick.color': '#000000'
})
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = "cm"

from main import *

# 定义参数范围
Js = np.arange(-0.30, 1.01, 0.05).round(2)
Ks = np.arange(-1.00, 0.31, 0.05).round(2)
d0s = [np.inf]

Ks = np.array([-0.00 if x == 0.00 else x for x in Ks])

# 创建模型列表
models = [
    ShortRangePhaseInter(
        K=K, J=J, d0=d0, 
        tqdm=True, savePath="./data", overWrite=True
    ) 
    for J, K, d0 in product(Js, Ks, d0s)
]


# 初始化一个列表来存储每个模型的序参量
order_parameters = []

# 遍历每个模型并计算序参量
for model in models:
    sa = StateAnalysis(model)
    model.positionX, model.phaseTheta = sa.get_state(-1)
    R = StateAnalysis.calc_order_parameter_R(model)
    S = StateAnalysis.calc_order_parameter_S(model)
    
    # 将结果存储为一个字典
    order_parameters.append({
        "J": model.J,
        "K": model.K,
        "d0": model.d0,
        "R": R,
        "S": S,
    })


# 将序参量结果打印出来
for params in order_parameters:
    print(f"J: {params['J']}, K: {params['K']}, d0: {params['d0']}")
    print(f"  R: {params['R']:.4f}")
    print(f"  S: {params['S']:.4f}")
    print("-" * 30)

df = pd.DataFrame(order_parameters)

# 打印 DataFrame 查看内容
print(df)

# 或者保存为 csv 文件
df.to_csv("order_parameters.csv", index=False)