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
    "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
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

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


class TwoOsillators(Swarmalators2D):
    def __init__(self, strengthLambda: float, r0: float,
                 typeA: str = "heaviside", 
                 omega1: float = 3, omega2: float = -3, dt: float=0.01, couplesNum: int=2,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 2, 
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        assert couplesNum in [1, 2]
        assert typeA in ["heaviside", "distanceWgt"]

        agentsNum = 2
        np.random.seed(randomSeed)
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.omegaTheta = np.array([omega1, omega2])
        radius = 3 / np.abs(self.omegaTheta)
        spatialAngle = self.phaseTheta - np.sign(self.omegaTheta) * np.pi / 2
        self.positionX = np.array([
            radius * np.cos(spatialAngle), radius * np.sin(spatialAngle)
        ]).T
        self.agentsNum = agentsNum
        self.couplesNum = couplesNum
        self.dt = dt
        self.speedV = 3
        self.r0 = r0
        if omega1 * omega2 > 0:
            if typeA == "heaviside":
                self.positionX[0] = self.positionX[0] + 3 / np.abs(omega1) * 2
            else:
                self.positionX[0] = self.positionX[0] - 3 / np.abs(omega1)

        self.typeA = typeA
        self.strengthLambda = strengthLambda
        self.tqdm = tqdm
        self.savePath = savePath
        self.temp = np.zeros(agentsNum)
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.randomSeed = randomSeed
        self.overWrite = overWrite

    def plot(self, ax: plt.Axes = None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        plt.quiver(
            self.positionX[:self.agentsNum // 2, 0], self.positionX[:self.agentsNum // 2, 1],
            np.cos(self.phaseTheta[:self.agentsNum // 2]), np.sin(self.phaseTheta[:self.agentsNum // 2]), color='tomato'
        )
        plt.quiver(
            self.positionX[self.agentsNum // 2:, 0], self.positionX[self.agentsNum // 2:, 1],
            np.cos(self.phaseTheta[self.agentsNum // 2:]), np.sin(self.phaseTheta[self.agentsNum // 2:]), color='dodgerblue'
        )
        plt.xlim(0, 10)
        plt.ylim(0, 10)

    @property
    def A(self):
        if self.typeA == "heaviside":
            rawA = self.distance_x(self.deltaX) <= self.r0
        elif self.typeA == "distanceWgt":
            rawA = np.exp(-self.distance_x(self.deltaX) / self.r0)
        if self.couplesNum == 2:
            return rawA
        else:
            rawA[0] = False
            return rawA

    @property
    def pointTheta(self):
        return self._pointTheta(self.phaseTheta, self.omegaTheta, self.strengthLambda, self.dt, self.A)

    @staticmethod
    @nb.njit
    def _pointTheta(phaseTheta: np.ndarray, omegaTheta: np.ndarray, strengthLambda: float, 
                    h: float, A: np.ndarray):
        adjMatrixTheta = np.repeat(phaseTheta, phaseTheta.shape[0]).reshape(phaseTheta.shape[0], phaseTheta.shape[0])
        k1 = omegaTheta + strengthLambda * np.sum(A * np.sin(
            adjMatrixTheta - phaseTheta
        ), axis=0)
        return k1 * h

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="pointTheta", value=pd.DataFrame(self.temp))

    def update(self):
        self.positionX[:, 0] += self.speedV * np.cos(self.phaseTheta) * self.dt
        self.positionX[:, 1] += self.speedV * np.sin(self.phaseTheta) * self.dt
        self.temp = self.pointTheta
        self.phaseTheta += self.temp
        # self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi
        # self.phaseTheta = np.mod(self.phaseTheta, 2 * np.pi)

    @property
    def deltaX(self) -> np.ndarray:
        return self.positionX - self.positionX[:, np.newaxis]

    def __str__(self) -> str:
        
        name =  (
            f"TwoOsillators_{self.typeA}_o1.{self.omegaTheta[0]}_o2.{self.omegaTheta[1]}_"
            f"{self.strengthLambda:.3f}_{self.r0:.2f}_{self.randomSeed}_c{self.couplesNum}"
        )
        
        return name

    def close(self):
        if self.store is not None:
            self.store.close()


class StateAnalysis:
    def __init__(self, model: TwoOsillators, classDistance: float = 2, lookIndex: int = -1, showTqdm: bool = False):
        self.model = model
        self.classDistance = classDistance
        self.lookIndex = lookIndex
        self.showTqdm = showTqdm
        
        targetPath = f"{self.model.savePath}/{self.model}.h5"
        totalPositionX = pd.read_hdf(targetPath, key="positionX")
        totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
        totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
        
        TNum = totalPositionX.shape[0] // self.model.agentsNum
        self.TNum = TNum
        self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
        self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)
        self.totalPointTheta = totalPointTheta.values.reshape(TNum, self.model.agentsNum)

        self.centersValue = None
        self.classesValue = None

        if self.showTqdm:
            self.iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            self.iterObject = range(1, self.totalPhaseTheta.shape[0])

    def get_state(self, index: int = -1):
        positionX = self.totalPositionX[index]
        phaseTheta = self.totalPhaseTheta[index]
        pointTheta = self.totalPointTheta[index]

        return positionX, phaseTheta, pointTheta