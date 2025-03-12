import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import os
import shutil

randomSeed = 100

# %matplotlib inline
# %config InlineBackend.figure_format = "svg"

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
plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"

from main import SpatialGroups, CorrectCouplingAfter, SingleDistribution
from multiprocessing import Pool

def run_model(model):
    # enhancedLambdas = np.linspace(0.009, 0.1, 50000)
    # model.run(enhancedLambdas)
    model.run(120000)

# randomSeeds = [10]
# rangeLambdas = np.concatenate([
#     np.arange(0.01, 0.1, 0.005), np.arange(0.1, 0.31, 0.05)
# ])
rangeLambdas = [0.3]
# rangeLambdas = np.arange(0.3, 1.01, 0.1)
# distanceDs = np.concatenate([
#     np.arange(0.1, 1.1, 0.05)
# ])
distanceDs = [2]

SAVE_PATH = r"E:\MS_ExperimentData\general"

models = [
    SingleDistribution(
        strengthLambda=l, distanceD0=d, boundaryLength=10, agentsNum=500, shotsnaps=100,
        savePath=SAVE_PATH, distributType="uniform", randomSeed=10, overWrite=True, tqdm=True) 
    for l in rangeLambdas
    for d in distanceDs
]
# models = [
#     CorrectCouplingAfter(strengthLambda=0.02, distanceD0=0.4, dt=0.01, tqdm=False, savePath="./data", uniform=True, randomSeed=seed, overWrite=False)
#     for seed in randomSeeds
# ]

run_model(models[0])

# with Pool(23) as p:
#     _ = list(tqdm(p.imap(run_model, models), total=len(models)))