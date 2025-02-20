import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import sys
import os
import shutil
from main import ChiralActiveMatterNonreciprocalReact
from main import draw_mp4

randomSeed = 10

# %matplotlib inline
# %config InlineBackend.figure_format = "retina"

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

import seaborn as sns
import matplotlib.font_manager as fm

if __name__ == "__main__":
      
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
    plt.rcParams['animation.ffmpeg_path'] = "D:/ffmpeg/bin/ffmpeg.exe"
    Lambdas = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    D0Means_range = np.arange(0.05, 1.05, 0.05)
    D0Means = np.append(D0Means_range, 2)
    # ODs = {'uniform', 'normal', 'lorentzian'}

    models = [
        ChiralActiveMatterNonreciprocalReact(chiralNum = 1, 
                                             strengthLambda = Lambda, 
                                             distanceD0Mean = D0Mean, distanceD0Std=0.1, d0Distribution = 'uniform', 
                                             omegaDistribution = 'uniform', 
                                             tqdm=True, savePath="./data/", overWrite=True
        )
        for Lambda, D0Mean, in product(Lambdas, D0Means)
    ]
    with Pool(processes=3) as p:
        p.map(draw_mp4, models)