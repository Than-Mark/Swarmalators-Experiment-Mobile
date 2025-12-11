import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import subprocess
import imageio
import os
import shutil
import sys
sys.path.append("..")

from main import *

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)

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

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"

from multiprocessing import Pool
import pandas as pd


# SAVE_PATH = r"E:\MS_ExperimentData\general"
# MP4_PATH = r"E:\MS_ExperimentData\mp4"
# MP4_TEMP_PATH = r"E:\MS_ExperimentData\mp4_temp"

SAVE_PATH = r"D:\MS_ExperimentData\general"
MP4_PATH = r"D:\MS_ExperimentData\mp4"
MP4_TEMP_PATH = r"D:\MS_ExperimentData\mp4_temp"


def draw_frame(sa: StateAnalysis):
    idx = sa.index
    
    # fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    sa.plot_spatial(ax=None, colorsBy="phase")

    # xShift = 0.
    # plt.xlim(0 + xShift, sa.model.boundaryLength + xShift)
    # plt.ylim(0, sa.model.boundaryLength)
    # plt.xticks(
    #     np.arange(0 + xShift, sa.model.boundaryLength + xShift + 1),
    #     np.arange(0, sa.model.boundaryLength + 1))
    # plt.tick_params(length=3, direction="in")
    plt.xlim(4, 6)
    plt.ylim(4, 6)

    plt.savefig(os.path.join(MP4_TEMP_PATH, f"{idx}.png"), bbox_inches='tight', dpi=200)
    plt.close()


if __name__ == "__main__":

    model = PhaseLagPatternFormation(
        strengthK=20, distanceD0=1, phaseLagA0=0.6 * np.pi,
        # initPhaseTheta=np.zeros(1000), 
        omegaMin=0, deltaOmega=0,
        agentsNum=102, dt=0.001,
        tqdm=True, savePath=SAVE_PATH, shotsnaps=1, 
        randomSeed=9, overWrite=True
    )

    # model = PhaseLagPatternFormation1D(strengthK=20, distanceD0=1, phaseLagA0=0.6*np.pi, 
    #                                    dt=0.001,
    #                                    tqdm=True, savePath=SAVE_PATH, shotsnaps=10, 
    #                                    randomSeed=9, overWrite=True)

    sa = StateAnalysis(model)
    subSaList = list()
    for i in tqdm(range(0, sa.TNum), desc="Processing data"):
        subSa = StateAnalysis()
        subSa.totalPositionX = [sa.totalPositionX[i]]
        subSa.totalPhaseTheta = [sa.totalPhaseTheta[i]]
        subSa.model = sa.model
        subSa.index = i
        subSa.model = sa.model
        subSaList.append(subSa)

    if os.path.exists(MP4_TEMP_PATH):
        shutil.rmtree(MP4_TEMP_PATH)
    os.mkdir(MP4_TEMP_PATH)
    
    with Pool(10) as p:
        p.map(
            draw_frame,
            tqdm(subSaList, desc="Drawing frames", total=sa.TNum),
        )
    
    if os.path.exists(MP4_PATH + rf"\{model}.mp4"):
        os.remove(rf"{MP4_PATH}\{model}.mp4")
        
    import imageio.v3 as iio
    img = iio.imread(os.path.join(MP4_TEMP_PATH, "0.png"))
    print(img.shape)  # output: (height, width, channels)

    fps = 60
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', os.path.join(MP4_TEMP_PATH, "%d.png"),
        '-vf', f'scale={img.shape[1] // 2 * 2}:{img.shape[0] // 2 * 2}:flags=lanczos', 
        '-c:v', 'libx264',
        '-crf', '28',  # Adjust the quality (lower is better, range 18-28)
        '-pix_fmt', 'yuv420p',
        '-an',  # No audio
        rf"{MP4_PATH}/{model}.mp4"
    ]

    subprocess.run(ffmpeg_command)
