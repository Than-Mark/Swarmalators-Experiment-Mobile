import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from functools import partial
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import subprocess
import imageio
import os
import shutil

randomSeed = 100

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

from main import *
from multiprocessing import Pool
import pandas as pd

SAVE_PATH = r"E:\MS_ExperimentData\general"
MP4_PATH = r"E:\MS_ExperimentData\mp4"
MP4_TEMP_PATH = r"E:\MS_ExperimentData\mp4_temp"


def draw_frame(sa: StateAnalysis):
    idx = sa.index
    positionX, internalState, c = sa.get_state(-1)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    stateColors = cmap(internalState)

    for i in range(sa.model.agentsNum):
        axs[0].add_artist(plt.Circle(
            positionX[i], sa.model.diameter / 2 * 0.95, zorder=1, 
            # facecolor="#9BD5D5", edgecolor="black"
            facecolor=stateColors[i], edgecolor="black"
        ))
    sc = plt.scatter(np.full_like(internalState, -1), np.full_like(internalState, -1), 
                    c=sa.model.internalState, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(sc, ax=axs[0], label=r"Internal State ($s_i$)")
    
    for ax in axs:
        ax.set_xlim(0, sa.model.boundaryLength)
        ax.set_ylim(0, sa.model.boundaryLength)

        im = ax.imshow(
            c.T, cmap=cmap, 
            extent=(0, sa.model.boundaryLength, 0, sa.model.boundaryLength),
            origin="lower", alpha=1, zorder=0,
            vmin=sa.minC, vmax=sa.maxC
        )
        
    plt.colorbar(im, ax=axs[1], label=r"Chemical Concentration ($c$)")

    plt.tight_layout()
    plt.savefig(os.path.join(MP4_TEMP_PATH, f"{idx}.png"), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":

    pathShift = 25
    center = 100
    patternHalfLength = 20
    nodePosition = np.array([
        [100 - pathShift, 100], [100 + pathShift, 100]
    ])

    model = PathPlanningGSCD(
        nodePosition=nodePosition,
        # nodePosition=np.array([]),
        productRateBetac=1, decayRateKc=0.001, diffusionRateDc=0.5,
        # cUpperThres=1, 
        cDecayBase=1, cControlThres=0.1,
        noiseRateBetaDp=0., initialState=0., chemoAlphaC=-50000,
        diameter=3, repelPower=2, repCutOff=True, deltaSpread=True,
        cellNumInLine=200, boundaryLength=200, agentsNum=200,
        tqdm=True, dt=0.1, savePath=SAVE_PATH, shotsnaps=100, overWrite=True
    )


    sa = StateAnalysis(model)
    subSaList = list()
    for i in tqdm(range(0, sa.TNum), desc="Processing data"):
        subSa = StateAnalysis()
        subSa.totalPositionX = [sa.totalPositionX[i]]
        subSa.totalInternalState = [sa.totalInternalState[i]]
        subSa.totalC = [sa.totalC[i]]
        subSa.model = sa.model
        subSa.index = i
        subSa.maxC, subSa.minC = sa.maxC, sa.minC
        subSaList.append(subSa)

    if os.path.exists(MP4_TEMP_PATH):
        shutil.rmtree(MP4_TEMP_PATH)
    os.mkdir(MP4_TEMP_PATH)
    
    with Pool(8) as p:
        p.map(
            draw_frame,
            tqdm(subSaList, desc="Drawing frames", total=sa.TNum),
        )
    
    if os.path.exists(MP4_PATH + rf"\{model}.mp4"):
        os.remove(rf"{MP4_PATH}\{model}.mp4")
    fps = 30
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', os.path.join(MP4_TEMP_PATH, "%d.png"),
        '-vf', 'setpts=0.20*PTS',  
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        rf"{MP4_PATH}/{model}.mp4"
    ]

    subprocess.run(ffmpeg_command)