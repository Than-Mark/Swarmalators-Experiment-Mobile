import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from itertools import product
from typing import List
import pandas as pd
import numpy as np
import numba as nb
import imageio
import sys
import os
import shutil

randomSeed = 10

if "ipykernel_launcher.py" in sys.argv[0]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)

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
if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"

sys.path.append("..")
from template import Swarmalators2D


class ObsAvoid(Swarmalators2D):
    def __init__(self, strengthLambda: float, alpha: float, boundaryLength: float = 10, 
                 agentsNum: int=1000, dt: float=0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, 
                 uniform: bool = True, randomSeed: int = 10, overWrite: bool = False) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.speedV = 3
        self.alpha = alpha
        self.uniform = uniform
        self.strengthLambda = strengthLambda
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.temp = {"pointTheta": np.zeros(agentsNum)}
        self.boundaryLength = boundaryLength
        self.halfBoundaryLength = boundaryLength / 2
        self.randomSeed = randomSeed
        self.overWrite = overWrite
        self.one = np.ones((agentsNum, agentsNum))

    def plot(self, ax: plt.Axes = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        plt.quiver(
            self.positionX[:, 0], self.positionX[:, 1],
            np.cos(self.phaseTheta), np.sin(self.phaseTheta), color='C0'
        )
        plt.xlim(0, 10)
        plt.ylim(0, 10)

    @property
    def deltaX(self) -> np.ndarray:
        return self._delta_x(self.positionX, self.positionX[:, np.newaxis], 
                             self.boundaryLength, self.halfBoundaryLength)

    @staticmethod
    @nb.njit
    def _delta_x(positionX: np.ndarray, others: np.ndarray,
                 boundaryLength: float, halfBoundaryLength: float) -> np.ndarray:
        subX = positionX - others
        return positionX - (
            others * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) + 
            (others - boundaryLength) * (subX < -halfBoundaryLength) + 
            (others + boundaryLength) * (subX > halfBoundaryLength)
        )

    @property
    def pointTheta(self):
        return self._pointTheta(self.phaseTheta, self.strengthLambda, self.alpha,
                                self.disDisSquare, 
                                self.A, self.dt)
    
    @staticmethod
    @nb.njit
    def _pointTheta(phaseTheta: np.ndarray, strengthLambda: float, alpha: float, 
                    divDisSquare: np.ndarray, 
                    A: np.ndarray, h: float):
        otherTheta = np.repeat(phaseTheta, phaseTheta.shape[0]).reshape(phaseTheta.shape[0], phaseTheta.shape[0])
        k1 = strengthLambda * np.sum(
            A * (otherTheta - phaseTheta) * divDisSquare
        , axis=0)
        return k1 * h

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="pointTheta", value=pd.DataFrame(self.temp["pointTheta"]))

    @staticmethod
    @nb.njit
    def _phi(deltaX: np.ndarray) -> np.ndarray:
        return np.arctan2(deltaX[:, :, 1], deltaX[:, :, 0])

    @property
    def phi(self):
        return self._phi(self.temp["deltaX"])

    @staticmethod
    @nb.njit
    def _A(phaseTheta: np.ndarray, phiValue: np.ndarray, alpha: float) -> np.ndarray:
        return (phaseTheta - alpha < phiValue) & (phiValue < phaseTheta + alpha)

    @property
    def A(self):
        return self._A(self.phaseTheta, self.temp["phi"], self.alpha)

    def update_temp(self):
        self.temp["deltaX"] = self.deltaX
        self.temp["distanceX"] = self.distance_x(self.temp["deltaX"])
        self.temp["phi"] = self.phi

    def update(self):
        self.update_temp()
        self.positionX[:, 0] += self.speedV * np.cos(self.phaseTheta) * self.dt
        self.positionX[:, 1] += self.speedV * np.sin(self.phaseTheta) * self.dt
        self.positionX = np.mod(self.positionX, self.boundaryLength)
        self.disDisSquare = self.div_distance_power(numerator=self.one, power=2, dim=1)
        self.temp["pointTheta"] = self.pointTheta
        self.phaseTheta += self.temp["pointTheta"]
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi

    def __str__(self) -> str:
        
        if self.uniform:
            name =  f"ObsAvoid_l{self.strengthLambda:.3f}_a{self.alpha:.2f}_{self.randomSeed}"
        else:
            name =  f"ObsAvoid_l{self.strengthLambda:.3f}_a{self.alpha:.2f}_{self.randomSeed}"

        return name

    def close(self):
        if self.store is not None:
            self.store.close()


def plot_last(model: ObsAvoid, savePath: str = "./data", ):
    targetPath = f"{savePath}/{model}.h5"
    totalPositionX = pd.read_hdf(targetPath, key="positionX")
    totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
    totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
    TNum = totalPositionX.shape[0] // model.agentsNum
    totalPositionX = totalPositionX.values.reshape(TNum, model.agentsNum, 2)
    totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, model.agentsNum)
    totalPointTheta = totalPointTheta.values.reshape(TNum, model.agentsNum)
    positionX = totalPositionX[-1]
    phaseTheta = totalPhaseTheta[-1]

    plt.subplots(figsize=(5, 5))

    plt.quiver(
        positionX[:, 0], positionX[:, 1],
        np.cos(phaseTheta), np.sin(phaseTheta), color='C0'
    )
    plt.xlim(0, model.boundaryLength)
    plt.ylim(0, model.boundaryLength)
    plt.show()

def draw_mp4(model: ObsAvoid, savePath: str = "./data", mp4Path: str = "./mp4"):

    targetPath = f"{savePath}/{model}.h5"
    totalPositionX = pd.read_hdf(targetPath, key="positionX")
    totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
    totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
    TNum = totalPositionX.shape[0] // model.agentsNum
    totalPositionX = totalPositionX.values.reshape(TNum, model.agentsNum, 2)
    totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, model.agentsNum)
    totalPointTheta = totalPointTheta.values.reshape(TNum, model.agentsNum)

    def plot_frame(i):
        pbar.update(1)
        positionX = totalPositionX[i]
        phaseTheta = totalPhaseTheta[i]
        fig.clear()
        ax1 = plt.subplot(1, 1, 1)
        ax1.quiver(
            positionX[:, 0], positionX[:, 1],
            np.cos(phaseTheta), np.sin(phaseTheta), color='C0'
        )
        limShift = 0
        ax1.set_xlim(0 - limShift, model.boundaryLength + limShift)
        ax1.set_ylim(0 - limShift, model.boundaryLength + limShift)

    pbar = tqdm(total=TNum)
    fig, ax = plt.subplots(figsize=(5, 5))
    ani = ma.FuncAnimation(fig, plot_frame, frames=np.arange(0, TNum, 1), interval=50, repeat=False)
    ani.save(f"{mp4Path}/{model}.mp4", dpi=100)
    plt.close()

    pbar.close()