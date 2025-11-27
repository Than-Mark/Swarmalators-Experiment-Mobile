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
import sys
sys.path.append("..")

randomSeed = 10

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
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

SAVE_PATH = r"D:\PythonProject\System Theory\Periodical Potential\data"

gammas = np.linspace(0, 1, 11)
dampingRatios = [0.1, 1, 10]  # np.linspace(1e-5, 3, 21)
randomSeed = 10
strengthLambdas = np.linspace(0.1, 1.0, 5)
distanceDs = [1.0]  # np.linspace(0.1, 3, 7)
kappas = [0.25, 0.50]  # np.linspace(1e-5, 3, 21)  # [1.0]

models = [
    PeriodicalPotential(
        strengthLambda=strengthLambda, distanceD=distanceD, gamma=gamma, dampingRatio=dampingRatio, kappa=kappa, L=1.5, agentsNum=1000, 
boundaryLength=5, dt = 0.005, tqdm=True, savePath = SAVE_PATH, overWrite=True
    )
    for strengthLambda in strengthLambdas
    for distanceD in distanceDs
    for dampingRatio in dampingRatios
    for kappa in kappas
    for gamma in gammas
]

def _ensure_sa(model):
    path1 = f"{model.savePath}/{model}.h5"
    class_name = model.__class__.__name__
    path2 = f"{model.savePath}/{class_name}_{model.randomSeed}.h5"
    if (not os.path.exists(path1)) and (not os.path.exists(path2)):
        model.run(10)
    return StateAnalysis(model)

sas = [_ensure_sa(model) for model in tqdm(models)]

sa_map = {}
for sa in sas:
    key = (sa.model.strengthLambda, sa.model.gamma, sa.model.dampingRatio, sa.model.kappa)
    sa_map[key] = sa

for dampingRatio in dampingRatios:
    for kappa in kappas:
        fig, axs = plt.subplots(
            len(strengthLambdas), len(gammas),
            figsize=(len(gammas) * 4, len(strengthLambdas) * 4),
            squeeze=False
        )

        for i, strengthLambda in enumerate(strengthLambdas):
            for j, gamma in enumerate(gammas):
                sa = sa_map[(strengthLambda, gamma, dampingRatio, kappa)]
                ax = axs[i, j]
                sa.plot_spatial(ax, colorsBy="chiral", index=-1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    rf"$\lambda={strengthLambda:.2f}, \ \gamma={gamma:.2f}$",
                    fontsize=12, loc="left"
                )
                ax.set_aspect("equal")

        plt.tight_layout()
        os.makedirs("figs", exist_ok=True)
        plt.savefig(
            f"figs/{sa.model.__class__.__name__}_grid_lambda_gamma_"
            f"zeta{dampingRatio}_kappa{kappa}.pdf",
            bbox_inches="tight"
        )
        plt.close()
