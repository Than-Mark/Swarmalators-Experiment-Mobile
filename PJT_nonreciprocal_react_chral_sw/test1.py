# 这是一个初始化文件，用于导入常用的库和设置一些全局的参数
import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import numpy as np
import numba as nb
import imageio
import os
import shutil
from main import *

randomSeed = 100

%matplotlib inline
%config InlineBackend.figure_format = "retina"

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
)

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

import seaborn as sns

sns.set(font_scale=1.1, rc={
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

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['animation.ffmpeg_path'] = "D:/Program Files/ffmpeg/bin/ffmpeg.exe"

for i in range(0.01,0.96,0.05):
    for j in {'uniform','normal','lorentzian'}:
        model = ChiralActiveMatterWithNoise(strengthLambda = i, distanceD0=2, noiseRateAlpha=0, omegaTheta= 2, distribution = j,  tqdm=True, savePath="./data/", overWrite=True)
        model.run(5000)
        draw_mp4(model)
