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

SAVE_PATH = r"E:\MS_ExperimentData\general"


def run_model(model: PhaseLagPatternFormation):
    model.run(80000)


if __name__ == "__main__":
    phaseLags = np.linspace(-1, 1, 21) * np.pi
    omegaMins = [0]# [0.1]  # np.linspace(0.1e5, 3, 21)
    randomSeed = 10  # Done: [9]
    strengthLambda = 20
    distanceD0 = 1
    deltaOmega = 0  # Done: [1]

    models = [
        PhaseLagPatternFormationNoCounter(
            strengthK=strengthLambda, distanceD0=distanceD0, phaseLagA0=phaseLag,
            # initPhaseTheta=np.zeros(1000), 
            omegaMin=omegaMin, deltaOmega=deltaOmega, dt=0.001,
            tqdm=True, savePath=SAVE_PATH, shotsnaps=10, 
            randomSeed=randomSeed, overWrite=False
        )
        for omegaMin in omegaMins
        for phaseLag in phaseLags
    ]

    with Pool(7) as p:
        p.map(
            run_model, 
            tqdm(models, desc="run models", total=len(models))
        )