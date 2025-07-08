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

# SAVE_PATH = "/home/thanmark/MS_DATA/cryst"
SAVE_PATH = r"D:\MS_ExperimentData\general"


def run_model(model: PhaseLagPatternFormation):
    # model.run(40000)
    model.run(80000)
    # model.run(160000)
    # model.run(320000)
    # model.run(640000)


if __name__ == "__main__":
    # phaseLags = np.linspace(-1, 1, 21) * np.pi
    phaseLags = [0.6 * np.pi]
    omegaMins = [0]  # np.linspace(1e-5, 3, 21)
    randomSeeds = range(10)
    # strengthKs = np.linspace(4, 20, 7)  # [20]  # np.linspace(1, 20, 7)
    strengthKs = [20]
    # distanceD0s = np.linspace(0.3, 1.1, 7)  #  np.linspace(0.1, 3, 7)  # [1]
    distanceD0s = [0.56666666]
    deltaOmegas = [1]  # np.linspace(1e-5, 3, 21)  # [1.0]

    models = [
        PhaseLagPatternFormation(
            strengthK=strengthK, distanceD0=distanceD0, phaseLagA0=phaseLag,
            freqDist="uniform", initPhaseTheta=None,
            omegaMin=omegaMin, deltaOmega=deltaOmega, 
            agentsNum=2000, dt=0.005,
            tqdm=True, savePath=SAVE_PATH, shotsnaps=10, 
            randomSeed=randomSeed, overWrite=False
        )
        for strengthK in strengthKs
        for distanceD0 in distanceD0s
        for omegaMin in omegaMins
        for deltaOmega in deltaOmegas
        for phaseLag in phaseLags
        for randomSeed in randomSeeds
    ]

    with Pool(min(len(models), 13)) as p:
    # with Pool(42) as p:
        p.map(
            run_model, 
            tqdm(models, desc="run models", total=len(models))
        )