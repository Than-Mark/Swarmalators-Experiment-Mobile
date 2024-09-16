import matplotlib.patches as patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ma
from tqdm import tqdm
from itertools import product
from main import ThreeBody

def plot_last(model: ThreeBody, alphaRate: float = 0.7):

    targetPath = f"./data/{model}.h5"
    class1, class2 = (
        np.concatenate([np.ones(model.agentsNum // 2), np.zeros(model.agentsNum // 2)]).astype(bool), 
        np.concatenate([np.zeros(model.agentsNum // 2), np.ones(model.agentsNum // 2)]).astype(bool)
    )
    totalPositionX = pd.read_hdf(targetPath, key="positionX")
    totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
    totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
    TNum = totalPositionX.shape[0] // model.agentsNum
    totalPositionX = totalPositionX.values.reshape(TNum, model.agentsNum, 2)
    totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, model.agentsNum)
    totalPointTheta = totalPointTheta.values.reshape(TNum, model.agentsNum)
    fig, ax = plt.subplots(figsize=(11, 5))
    plt.title(f"{model}")
    idx = -76
    positionX = totalPositionX[idx]
    phaseTheta = totalPhaseTheta[idx]
    # class1, class2 = np.where(pointTheta > 0), np.where(pointTheta < 0)

    ax1 = plt.subplot(1, 2, 1)
    ax1.quiver(
        positionX[class1, 0], positionX[class1, 1],
        np.cos(phaseTheta[class1]), np.sin(phaseTheta[class1]), color='red', alpha=(1 - alphaRate) + (np.abs(model.omegaTheta[class1]) - 1) / 2 * alphaRate
    )
    ax1.quiver(
        positionX[class2, 0], positionX[class2, 1],
        np.cos(phaseTheta[class2]), np.sin(phaseTheta[class2]), color='blue', alpha=(1 - alphaRate) + (np.abs(model.omegaTheta[class2]) - 1) / 2 * alphaRate
    )
    ax1.spines['bottom'].set_color('black')
    ax1.spines['top'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.set_xticks([0, 5, 10])
    ax1.set_yticks([0, 5, 10])
    ax1.set_xlabel(r"x")
    ax1.set_ylabel(r"y")
    ax1.grid(False)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)

    ax2 = plt.subplot(1, 2, 2)
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    ax2.add_artist(circle)
    ax2.scatter(
        np.cos(phaseTheta[class1]), np.sin(phaseTheta[class1]), color='red', alpha=(1 - alphaRate) + (np.abs(model.omegaTheta[class1]) - 1) / 2 * alphaRate
    )
    ax2.scatter(
        np.cos(phaseTheta[class2]), np.sin(phaseTheta[class2]), color='blue', alpha=(1 - alphaRate) + (np.abs(model.omegaTheta[class2]) - 1) / 2 * alphaRate
    )
    lim = 1.2
    ax2.add_patch(patches.FancyArrowPatch(
        (0, -lim), (0, lim),
        color='black', arrowstyle='->', mutation_scale=15
    ))
    ax2.add_patch(patches.FancyArrowPatch(
        (-lim, 0), (lim, 0),
        color='black', arrowstyle='->', mutation_scale=15
    ))
    ax2.text(0.1, 0.1, r"$O$", ha="center", va="center", fontsize=16)
    
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.grid(False)
    ax2.set_yticks([-1., -0.5, 0., 0.5, 1.])
    ax2.set_xticks([-1., -0.5, 0., 0.5, 1.])
    ax2.set_xlabel(r"$\cos\varphi_i$")
    ax2.set_ylabel(r"$\sin\varphi_i$")

    plt.savefig(f"./figs/{model}_2d_circle.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    

    rangeLambdas = np.concatenate([
        np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.2)
    ])
    distanceDs = np.concatenate([
        np.arange(0.1, 1, 0.2)
    ])

    models = [
        ThreeBody(l1, l2, d1, d2, agentsNum=200, boundaryLength=5,
                tqdm=True, savePath="./data", overWrite=False)
        for l1, l2, d1, d2  in product(rangeLambdas, rangeLambdas, distanceDs, distanceDs)
    ]

    for model in tqdm(models):
        try:
            plot_last(model)
        except Exception as e:
            # print(e)
            continue