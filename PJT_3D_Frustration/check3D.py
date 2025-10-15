from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from matplotlib import gridspec
from tqdm.notebook import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import pickle
import json
import os
import shutil
import sys
sys.path.append("..")

from main import *

randomSeed = 100

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

with open("../swarmalatorlib/hex_colors.json", "r", encoding="utf-8") as f:
    hexColors = json.load(f)
hexCmap = mcolors.LinearSegmentedColormap.from_list("cmap", hexColors)


import seaborn as sns

sns.set_theme(
    style="ticks", 
    font_scale=1.1, rc={
    'figure.figsize': (6, 5),
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.5,
    "lines.linewidth": 1.5,
    'axes.grid': True,
    'text.color': '#000000',
    'figure.titleweight': "bold",
    'xtick.color': '#000000',
    'ytick.color': '#000000',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.right': False,
    'axes.spines.top': False,
})

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
#plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"

from multiprocessing import Pool
import pandas as pd

colors = ["#403990", "#3A76D6", "#FFC001", "#F46F43", "#FF0000"]
cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors)
cmap_r = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors[::-1])

colors = ["#403990", "#80A6E2", "#F46F43", "#CF3D3E"]
cmap2 = mcolors.LinearSegmentedColormap.from_list("cmap2", colors)

SAVE_PATH = r"F:\MS_ExperimentData\general"
MP4_PATH = r"F:\MS_ExperimentData\mp4"

# SAVE_PATH = r"D:\MS_ExperimentData\general"
# MP4_PATH = r"D:\MS_ExperimentData\mp4"

# LOCAL_FIG_PATH = "./PCT_C_figs"
LOCAL_FIG_PATH = "./figs"


from matplotlib.colors import Normalize

def vectors_to_cmap_colors(vectors, theta_weight=0.8):
    """
    将单位向量映射到 colormap 的颜色（结合 θ 和 φ）。
    
    参数：
        vectors: (N, 3) 数组，单位向量。
        cmap_name: matplotlib colormap 名称。
        theta_weight: θ 的权重（0 表示仅用 φ，1 表示 φ 和 θ 同等重要）。
    
    返回：
        colors: (N, 4) RGBA 数组。
    """
    x, y, z = vectors.T
    theta = np.arccos(z)  # [0, π]
    phi = np.arctan2(y, x)  # [-π, π]
    phi[phi < 0] += 2 * np.pi  # [0, 2π)
    
    # 归一化
    phi_norm = phi / (2 * np.pi)  # [0,1]
    theta_norm = theta / np.pi  # [0,1]
    
    # 结合 φ 和 θ
    scalar = (1 - theta_weight) * phi_norm + theta_weight * theta_norm
    # print(scalar)
    # 使用 colormap 映射
    norm = Normalize(vmin=0, vmax=1)
    colors = hexCmap(norm(scalar))
    return colors


model = Frustration3D(strengthA=1, strengthB=-0.5, 
                      distanceD0=5, phaseLagAlpha0=0.7 * np.pi,
                      boundaryLength=10, speedV=3, agentsNum=1000,
                      tqdm=True, savePath=SAVE_PATH, 
                      shotsnaps=5, dt=0.01,
                      randomSeed=10, overWrite=False)
positionX = pd.read_hdf(f"temp/SingleLastState_{model}.h5", key="positionX").values
phaseSigma = pd.read_hdf(f"temp/SingleLastState_{model}.h5", key="phaseSigma").values
colors = vectors_to_cmap_colors(phaseSigma)

fig = plt.figure(figsize=(10, 10))
ax: Axes3D = fig.add_subplot(111, projection="3d")

ax.scatter(positionX[:, 0], positionX[:, 1], positionX[:, 2], c=colors, s=5)
ax.set_box_aspect([1,1,1])
ax.set_xlim(0, model.boundaryLength)
ax.set_ylim(0, model.boundaryLength)
ax.set_zlim(0, model.boundaryLength)
plt.show()