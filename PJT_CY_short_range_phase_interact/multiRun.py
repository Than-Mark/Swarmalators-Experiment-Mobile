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

def run_model(model):
        model.run(10)


if __name__ == "__main__":

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

    # 扫描的参数范围
    Js = [0.5]
    Ks = np.arange(-1, 0.21, 0.1).round(2)
    d0s = [np.inf]

    models = [
        ShortRangePhaseInter(
            K=K, J=J, d0=d0, 
            tqdm=True, savePath="./data", overWrite=True
        ) 
        for J, K, d0 in product(Js, Ks, d0s)
    ]

    # processes为进程数，表示同时执行的进程数，可以根据CPU核数进行设置
    with Pool(processes=4) as p:
        p.map(run_model, models)