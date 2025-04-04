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

SAVE_PATH = "data"
MP4_PATH = "mp4"
MP4_TEMP_PATH = "mp4_temp"


def draw_frame(sa: StateAnalysis):
    idx = sa.index
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [5, 8.5]})
    sa.plot_spatial(ax=ax1, index=-1)
    sa.plot_fields(ax=ax2, index=-1, fixExtremum=False)
    plt.tight_layout()
    plt.savefig(os.path.join(MP4_TEMP_PATH, f"{idx}.png"))
    plt.close(fig)


# k23 = 0.4
# model = ChemotacticLotkaVolterra(
#     k1=0.01, k2=k23, k3=k23, k4=0.01,
#     boundaryLength=20, speedV=0.0, 
#     diameter=0.3, repelPower=1,
#     omega1=0, omega2=0, filedDrive=False,
#     cellNumInLine=200, agentsNum=1000,
#     chemoAlpha1=-1, chemoAlpha2=-1,
#     diffusionRateD1=0.01, diffusionRateD2=0.01,
#     dt=0.01, shotsnaps=200,
#     tqdm=True, savePath=SAVE_PATH, overWrite=True
# )

def get_sub_sa_list(model: ChemotacticLotkaVolterra):
    sa = StateAnalysis(model)
    subSaList = list()
    for i in range(0, sa.TNum):
        subSa = StateAnalysis()
        subSa.totalPositionX = [sa.totalPositionX[i]]
        subSa.totalPhaseTheta = [sa.totalPhaseTheta[i]]
        subSa.totalDotPos = [sa.totalDotPos[i]]
        subSa.totalC1 = [sa.totalC1[i]]
        subSa.totalC2 = [sa.totalC2[i]]
        subSa.model = sa.model
        subSa.index = i
        subSa.maxC1, subSa.maxC2 = sa.maxC1, sa.maxC2
        subSa.minC1, subSa.minC2 = sa.minC1, sa.minC2
        subSaList.append(subSa)

    return subSaList


def generate_mp4(model: ChemotacticLotkaVolterra):
    sa = StateAnalysis(model)
    subSaList = get_sub_sa_list(model)

    if os.path.exists(MP4_TEMP_PATH):
        shutil.rmtree(MP4_TEMP_PATH)
    os.mkdir(MP4_TEMP_PATH)
    
    print(f"Generating mp4 for model {model}")
    with Pool(30) as p:
        p.map(
            draw_frame,
            tqdm(subSaList, desc="Drawing frames", total=sa.TNum),
        )

    print("Generating mp4")
    if os.path.exists(rf"{MP4_PATH}/{model}.mp4"):
        os.remove(rf"{MP4_PATH}/{model}.mp4")
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


if __name__ == "__main__":

    # k1 = 0.19
    # k4 = 0.19
    # k1s = [k1]
    # k4s = [k4]
    # k1s = np.arange(0.01, 0.32, 0.06)
    # k4s = np.arange(0.01, 0.32, 0.06)
    k1s = np.arange(0.01, 0.08, 0.02)
    k4s = np.arange(0.01, 0.32, 0.07)
    k23 = 0.5
    diffusionRateD1s = diffusionRateD2s = [0.01]
    chemoAlpha1s = chemoAlpha2s = [-1]
    # chemoAlpha1s = [-1.67]
    # chemoAlpha2s = [-0.01]


    models = [
        ChemotacticLotkaVolterra(
            k1=k1, k2=k23, k3=k23, k4=k4,
            boundaryLength=20, speedV=0.0, 
            diameter=0.3, repelPower=2,
            omega1=0, omega2=0, fieldDrive=True,
            cellNumInLine=200, agentsNum=1000,
            chemoAlpha1=a1, chemoAlpha2=a2,
            diffusionRateD1=D1, diffusionRateD2=D2,
            dt=0.01, shotsnaps=100,
            tqdm=True, savePath=SAVE_PATH, overWrite=True
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
    # models = [
    #     ChemotacticLotkaVolterra(
    #         k1=0.01, k2=k23, k3=k23, k4=0.29,
    #         boundaryLength=20, speedV=0.01, 
    #         diameter=0.3, repelPower=2,
    #         omega1=0, omega2=0, fieldDrive=True,
    #         cellNumInLine=200, agentsNum=1000,
    #         chemoAlpha1=-1, chemoAlpha2=-1,
    #         diffusionRateD1=0.01, diffusionRateD2=0.01,
    #         dt=0.01, shotsnaps=100,
    #         tqdm=True, savePath=SAVE_PATH, overWrite=True
    #     )
    # ]

    for model in models:
        generate_mp4(model)