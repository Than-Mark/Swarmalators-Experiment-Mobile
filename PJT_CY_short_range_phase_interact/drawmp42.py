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



# 定义一个包裹函数，用于多进程中调用 draw_mp4
def generate_video_wrapper(args):
    J, K, d0 = args
    model = ShortRangePhaseInter(K=K, J=J, d0=d0, tqdm=False, savePath="./data", overWrite=True)
    draw_mp4(model, savePath="./data", mp4Path="./mp4")


def main():
    # 参数范围设置
    Js = [-0.1, 0, 0.1, 0.5, 1]
    Ks = [-1, -0.5, -0.2, -0.1, 0, 0.1]
    d0s = [0.1, 0.2, 0.3, 0.5]

    # 使用多进程池并行处理任务
    params = list(product(Js, Ks, d0s))

    # 创建保存路径
    os.makedirs('./mp4', exist_ok=True)

    # 使用 Pool 进行多进程并行处理
    with Pool(processes=5) as pool:
        list(tqdm(pool.imap(generate_video_wrapper, params), total=len(params)))


if __name__ == "__main__":
    main()
