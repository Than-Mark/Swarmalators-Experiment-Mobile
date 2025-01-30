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

from main import *
from multiprocessing import Pool
import pandas as pd

SAVE_PATH = r"E:\MS_ExperimentData\general"
MP4_PATH = r"E:\MS_ExperimentData\mp4"
MP4_TEMP_PATH = r"E:\MS_ExperimentData\mp4_temp"


def draw_frame(sa: StateAnalysis, idx: int):
    fig = plt.figure(figsize=(13.5, 4))
    ax1 = fig.add_subplot(131)
    sa.plot_spatial(ax=ax1, index=idx)
    ax2 = fig.add_subplot(132)
    im = ax2.pcolor(sa.totalC1[idx].T, cmap=cmap, vmin=0, vmax=sa.maxC1)
    plt.colorbar(im, ax=ax2, cmap=cmap)
    ax3 = fig.add_subplot(133)
    im = ax3.pcolor(sa.totalC2[idx].T, cmap=cmap, vmin=0, vmax=sa.maxC2)
    plt.colorbar(im, ax=ax3, cmap=cmap)
    ax1.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(MP4_TEMP_PATH, f"{idx}.png"))
    plt.close(fig)


model = ChemotacticLotkaVolterra(
    k1=0.01, k2=0.5, k3=0.5, k4=0.01,
    boundaryLength=10, speedV=0.1, 
    diameter=0.4, repelPower=1,
    cellNumInLine=100, agentsNum=200,
    chemoAlpha1=-5, chemoAlpha2=-5,
    diffusionRateD1=0.01, diffusionRateD2=0.01,
    dt=0.1, shotsnaps=5,
    tqdm=True, savePath=SAVE_PATH, overWrite=True
)
sa = StateAnalysis(model)

if __name__ == "__main__":

    if os.path.exists(MP4_TEMP_PATH):
        shutil.rmtree(MP4_TEMP_PATH)
    os.mkdir(MP4_TEMP_PATH)
    
    draw_frame_with_sa = partial(draw_frame, sa)
    with Pool(10) as p:
        p.map(
            draw_frame_with_sa,
            tqdm(range(0, sa.TNum), desc="Drawing frames", total=sa.TNum),
        )
    
    if os.path.exists(MP4_PATH + rf"\{model}.mp4"):
        os.remove(rf"{MP4_PATH}\{model}.mp4")
    fps = 30
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', os.path.join(MP4_TEMP_PATH, "%d.png"),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        rf"{MP4_PATH}\{model}.mp4"
    ]

    subprocess.run(ffmpeg_command)