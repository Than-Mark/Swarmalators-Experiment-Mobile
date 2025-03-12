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

alphas = np.concatenate([
    np.arange(0.1, 1, 0.05), np.arange(1, 2.1, 0.5)
])
chemotacticStrengthBetaRs = np.array([0.1, 0.5, 1, 2, 5, 10])
diffusionRates = np.arange(0.002, 0.021, 0.002)

# SAVE_PATH = "./data"
SAVE_PATH = r"E:\MS_ExperimentData\general"

# models = [
#     GSPatternFormation(strengthLambda=0.1, alpha=a, 
#         boundaryLength=10, cellNumInLine=250, 
#         productRateUK0=1, productRateVK0=1,
#         decayRateKd=0.001, 
#         chemoBetaU=b, chemoBetaV=-b,
#         diffusionRateDc=0.002, epsilon=10, 
#         c0=0.012, dt=0.02, shotsnaps=5,
#         tqdm=True, savePath="./data", overWrite=True)
#     for a, b in tqdm(
#         product(alphas, chemotacticStrengthBetaRs), 
#         desc="models",
#         total=len(alphas) * len(chemotacticStrengthBetaRs)
#     )
# ]
# beta = -5
# models = [
#     GSPatternFormation(strengthLambda=0, alpha=1, 
#         boundaryLength=10, cellNumInLine=250, 
#         productRateUK0=1, productRateVK0=1,
#         decayRateKd=0.001, 
#         chemoBetaU=beta, chemoBetaV=beta,
#         diffusionRateDu=Du, diffusionRateDv=Dv,
#         dt=0.02, shotsnaps=20, typeA="heaviside",
#         distribution="uniform", omegaMean=2, omegaStd=0,
#         tqdm=True, savePath="./data", overWrite=True)
#     for Du, Dv in tqdm(
#         product(diffusionRates, diffusionRates), 
#         desc="models",
#         total=len(diffusionRates) ** 2
#     )
# ]

k1s = np.arange(0.01, 0.32, 0.03)
k4s = np.arange(0.01, 0.32, 0.03)
k23 = 0.5
# k1 = 0.1
# k4 = 0.4
# k1s = [k1]
# k4s = [k4]
# diffusionRateD1s = np.arange(0.001, 0.01, 0.001)
# diffusionRateD2s = np.arange(0.001, 0.01, 0.001)
diffusionRateD1s = diffusionRateD2s = [0.01]
# chemoAlpha1s = -np.linspace(1, 5, 9)
# chemoAlpha2s = np.linspace(1, 5, 9)
chemoAlpha1s = chemoAlpha2s = [-5]


models = [
    ChemotacticLotkaVolterra(
        k1=k1, k2=k23, k3=k23, k4=k4,
        boundaryLength=10, speedV=0.1, 
        diameter=0.4, repelPower=1,
        cellNumInLine=100, agentsNum=200,
        chemoAlpha1=a1, chemoAlpha2=a2,
        diffusionRateD1=D1, diffusionRateD2=D2,
        dt=0.1, shotsnaps=5,
        tqdm=True, savePath=SAVE_PATH, overWrite=False
    )
    for k1, k4, D1, D2, a1, a2 in tqdm(
        list(product(
            k1s, k4s, 
            diffusionRateD1s, diffusionRateD2s, 
            chemoAlpha1s, chemoAlpha2s
        )),
        desc="models",
    )
]

if __name__ == "__main__":

    with Pool(10) as p:
        p.map(
            run_model, 
            tqdm(models, desc="run models", total=len(models))
        )