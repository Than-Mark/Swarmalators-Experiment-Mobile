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


Js = [1, 0.5, 0.1, 0, -0.1]
Ks = [-1, -0.5, -0.2, -0.1, 0, 0.1]
d0 = 0.1

#fig, axs = plt.subplots(len(Ks), len(Js), figsize=(len(Ks) * 4, len(Js) * 4))
fig, axs = plt.subplots(len(Js), len(Ks), figsize=(27, 23))
axs = axs.flatten()
idx = 0

for J, K in tqdm(list(product(Js, Ks))):
    model = ShortRangePhaseInter(K=K, J=J, d0=d0, tqdm=False, savePath="./data", overWrite=False)
    sa = StateAnalysis(model)
    ax = axs[idx]
    step = 6000
    sa.plot_state_at_step(ax=ax, step=step, withColorBar=False)
    ax.set_title(f"J={J}, K={K}")
    idx += 1

plt.tight_layout()
plt.savefig(f"./figs/phase_diagram_d0_{d0}_step_{step}.png", dpi=100)