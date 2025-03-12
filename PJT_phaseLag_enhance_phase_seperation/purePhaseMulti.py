
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

def run_model(model):
        model.run(10000)

if __name__ == "__main__":

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
    )

    @nb.njit
    def colors_idx(phaseTheta):
        return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"

    from main import *
    from multiprocessing import Pool

    omegaMins = [0.1]  # np.linspace(0.1, 0.5, 30)
    phaseLags = np.linspace(-1, 1, 31) * np.pi
    randomSeed = 10

    strengthLambda = 0.15 * 32 * 2
    deltaOmega = 1

    # SAVE_PATH = "./data"  # 
    SAVE_PATH = r"D:\MS_ExperimentData\general"

    models = [
        PurePhaseModel(strengthLambda=strengthLambda,
                       phaseLag=phaseLag, 
                       distribution="cauchy", initPhaseTheta=np.zeros(1000),
                       agentsNum=1000,
                       omegaMin=omegaMin, deltaOmega=deltaOmega,
                       savePath=SAVE_PATH, dt=0.01,
                       tqdm=True, overWrite=True, randomSeed=randomSeed)
        for omegaMin in omegaMins
        for phaseLag in phaseLags
    ]

    # for model in models:
    #     print(str(model))

    with Pool(31) as p:
        # p.map(run_model, models)

        p.map(
            run_model,
            tqdm(models, desc="run models", total=len(models))
        )