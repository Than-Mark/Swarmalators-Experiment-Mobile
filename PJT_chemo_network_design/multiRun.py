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

def run_model(model):
    model.run(12000)

Aus = np.linspace(-6, 6, 3)
kus = np.linspace(-0.05, 0.05, 3)
productRateBetau = -0.1
productRateBetav = 0

# SAVE_PATH = "./data"
SAVE_PATH = r"E:\MS_ExperimentData\general"



pathShift = 7
center = 100
nodePosition = [
    [
        [center - pathShift * i, center - pathShift * i],
        [center - pathShift * i, center + pathShift * i],
        [center + pathShift * i, center - pathShift * i],
        [center + pathShift * i, center + pathShift * i]
    ]
    for i in range(0, 10)
]
nodePosition = np.unique(np.concatenate(nodePosition, axis=0), axis=0)

models = [
    PathPlanningGS(
        nodePosition=nodePosition,
        productRateBetau=productRateBetau, productRateBetav=productRateBetav,
        productRateKu=ku, productRateKv=0,
        u0=0.32, v0=0.25,
        decayRateKd=0.08, decayRateKf=0.03,
        diffusionRateDu=0.5, diffusionRateDv=0.25,
        chemoAlphaU=Au, chemoAlphaV=0,
        diameter=3, repelPower=2, repCutOff=True,
        boundaryLength=200, cellNumInLine=200,
        tqdm=True, savePath=SAVE_PATH, overWrite=True,
        dt=0.1, shotsnaps=50,
    )
    for ku, Au in tqdm(
        product(kus, Aus), 
        desc="models",
        total=len(kus) * len(Aus)
    )
]

if __name__ == "__main__":

    with Pool(9) as p:
        p.map(
            run_model, 
            tqdm(models, desc="run models", total=len(models))
        )