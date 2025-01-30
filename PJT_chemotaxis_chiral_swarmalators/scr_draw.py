import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import os
import shutil

randomSeed = 100

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)

colors = ["#5657A4", "#95D3A2", "#FFFFBF", "#F79051", "#A30644"]
cmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
cmap_r = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors[::-1])

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

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
    'text.color': '#000000',
    'figure.titleweight': "bold",
    'xtick.color': '#000000',
    'ytick.color': '#000000'
})

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"

from main import *
from multiprocessing import Pool
import pandas as pd

# SAVE_PATH = "./data"
SAVE_PATH = r"E:\MS_ExperimentData\general"

class1, class2 = (
    np.concatenate([np.ones(500), np.zeros(500)]).astype(bool), 
    np.concatenate([np.zeros(500), np.ones(500)]).astype(bool)
)
alphas = np.concatenate([
    np.arange(0.1, 1, 0.05), np.arange(1, 2.1, 0.5)
])
chemotacticStrengthBetaRs = np.array([0.1, 0.5, 1, 2, 5, 10])
diffusionRates = np.arange(0.002, 0.021, 0.002)
beta = -5

k1s = np.arange(0.01, 0.32, 0.03)
k4s = np.arange(0.01, 0.32, 0.03)
k23 = 0.5
# k1 = 0.1
# k4 = 0.4
# diffusionRateD1s = np.arange(0.001, 0.01, 0.001)
# diffusionRateD2s = np.arange(0.001, 0.01, 0.001)
D1 = D2 = 0.01
# chemoAlpha1s = -np.linspace(1, 5, 9)
# chemoAlpha2s = np.linspace(1, 5, 9)
a1 = a2 = -5

# Plot Big Graph
fig = plt.figure(figsize=(len(k1s) * 4, len(k4s) * 4))
# fig = plt.figure(figsize=(len(diffusionRateD1s) * 4, len(diffusionRateD2s) * 4))
# fig = plt.figure(figsize=(len(chemoAlpha1s) * 4, len(chemoAlpha2s) * 4))

idx = 1

for k1, k4 in tqdm(product(k1s, k4s), total=len(k1s) * len(k4s)):
# for D1, D2 in tqdm(product(diffusionRateD1s, diffusionRateD2s), total=len(diffusionRateD1s) * len(diffusionRateD2s)):
# for a1, a2 in tqdm(product(chemoAlpha1s, chemoAlpha2s), total=len(chemoAlpha1s) * len(chemoAlpha2s)):
    model = ChemotacticLotkaVolterra(
        k1=k1, k2=k23, k3=k23, k4=k4,
        boundaryLength=10, speedV=0.1, 
        diameter=0.4, repelPower=1,
        cellNumInLine=100, agentsNum=200,
        chemoAlpha1=a1, chemoAlpha2=a2,
        diffusionRateD1=D1, diffusionRateD2=D2,
        dt=0.1, shotsnaps=5,
        tqdm=True, savePath=SAVE_PATH, overWrite=True
    )

    sa = StateAnalysis(model)

    ax = plt.subplot(len(k1s), len(k4s), idx)
    # ax = plt.subplot(len(diffusionRateD1s), len(diffusionRateD2s), idx)
    # ax = plt.subplot(len(chemoAlpha1s), len(chemoAlpha2s), idx)
    sa.plot_spatial(ax=ax, index=-1)

    ax.set_xlim(0, model.boundaryLength)
    ax.set_ylim(0, model.boundaryLength)    
    ax.set_title(rf"$k_1={np.round(k1, 4)}, k_4={np.round(k4, 4)}$", fontsize=16)
    # ax.set_title(rf"$\alpha_1={np.round(a1, 4)}, \alpha_2={np.round(a2, 4)}$", fontsize=16)

    idx += 1

plt.tight_layout()
plt.savefig(f"bigGraphParticle_k23_{k23}.png", dpi=100, bbox_inches="tight")
# plt.savefig(f"bigGraphParticle_k1_{k1}_k23_{k23}_k4_{k4}.png", dpi=100, bbox_inches="tight")
plt.close()