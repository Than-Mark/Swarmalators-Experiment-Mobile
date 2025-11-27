import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ma
from tqdm import tqdm

savePath = "./data"
mp4Path = "./mp4"

def draw_mp4(model):
    
    targetPath = f"./data/{model}.h5"
    totalPositionX = pd.read_hdf(targetPath, key="positionX")
    totalSpeed = pd.read_hdf(targetPath, key="speed")
    totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
    totalPointThetaSpeed = pd.read_hdf(targetPath, key="pointThetaSpeed")
    TNum = totalPositionX.shape[0] // model.agentsNum
    totalPositionX = totalPositionX.values.reshape(TNum, model.agentsNum, 2)
    totalSpeed = totalSpeed.values.reshape(TNum, model.agentsNum, 2)
    totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, model.agentsNum)
    totalPointThetaSpeed = totalPointThetaSpeed.values.reshape(TNum, model.agentsNum)
    shift = 0
    class1, class2 = (
        np.concatenate([np.ones(model.agentsNum // 2), np.zeros(model.agentsNum // 2)]).astype(bool), 
        np.concatenate([np.zeros(model.agentsNum // 2), np.ones(model.agentsNum // 2)]).astype(bool)
    )

    def plot_frame(i):
        pbar.update(1)
        positionX = totalPositionX[i]
        phaseTheta = totalPhaseTheta[i]
        fig.clear()
        ax1 = plt.subplot(1, 2, 1)
        ax1.quiver(
            positionX[class1, 0], positionX[class1, 1],
            np.cos(phaseTheta[class1]), np.sin(phaseTheta[class1]), color='tomato'
        )
        ax1.quiver(
            positionX[class2, 0], positionX[class2, 1],
            np.cos(phaseTheta[class2]), np.sin(phaseTheta[class2]), color='dodgerblue'
        )
        limShift = 0
        ax1.set_xlim(0 - limShift, model.boundaryLength + limShift)
        ax1.set_ylim(0 - limShift, model.boundaryLength + limShift)

        ax2 = plt.subplot(1, 2, 2, projection='3d')
        hist, bins = np.histogram(phaseTheta[class1], bins=100, range=(-np.pi, np.pi))
        # print(np.array([np.zeros_like(hist), hist]).shape)
        ax2.plot_surface(
            np.cos(bins[:-1]), np.sin(bins[:-1]), 
            np.array([np.zeros_like(hist), hist]), 
            color='tomato', alpha=0.5, edgecolor="tomato"
        )
        hist, bins = np.histogram(phaseTheta[class2], bins=100, range=(-np.pi, np.pi))
        ax2.plot_surface(
            np.cos(bins[:-1]) + shift, np.sin(bins[:-1]) + shift,
            np.array([np.zeros_like(hist), hist]), 
            color='dodgerblue', alpha=0.5, edgecolor="dodgerblue"
        )
        ax2.set_xlabel(r"$\cos(\theta_I)$")
        ax2.set_ylabel(r"$\sin(\theta_I)$")
        ax2.set_zlabel("Count")
        ax2.set_zlim(0, 1000)

    pbar = tqdm(total=TNum)
    fig, ax = plt.subplots(figsize=(11, 5))
    ani = ma.FuncAnimation(fig, plot_frame, frames=np.arange(0, TNum, 1), interval=50, repeat=False)
    ani.save(f"{mp4Path}/{model}.mp4")
    plt.close()

    pbar.close()

if __name__ == "__main__":
    from itertools import product
    from main import PeriodicalPotential
    from multiprocessing import Pool

    rangeLambdas = np.concatenate([
        np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.2)
    ])
    distanceDs = np.concatenate([
        np.arange(0.1, 1, 0.2)
    ])
    rangeGamma = np.concatenate([
        np.arange(1.0, 11.0, 0.1)
    ])
    kappa = [3]
    period = [0.5]

    savePath = "./data"

    models = [
        PeriodicalPotential(l, d, g, k, T, agentsNum=200, boundaryLength=5,
                  tqdm=True, savePath=savePath, overWrite=False)
        for l, d, g, k, T in product(rangeLambdas, distanceDs, rangeGamma, kappa, period)
    ]

    with Pool(32) as p:
        p.map(draw_mp4, models)
