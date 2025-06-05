import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from typing import List, Tuple
from itertools import product
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


class ChemoSensingCAP(Swarmalators2D):
    def __init__(self, agentsNum: int = 1000, speedV: float = 0.1,
                 chemoAlpha: float = -1,
                 omegaMax: float = 0.1, omegaMin: float = -0.1, 
                 vOmega: float = 0.1, cTh: float = 0.5,
                 boundaryLength: float = 10, cellNumInLine: int = 200,
                 diffusionRateD: float = 1.0, feedRate: float = 0.001, 
                 decayRate: float = 0.001, consumptionRate: float = 1.0,
                 dt: float=0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10, 
                 randomSeed: int = 10, overWrite: bool = False):
        """
        make sure that dt/dx² ≤ 1/(2D)
        """

        self.agentsNum = agentsNum
        self.speedV = speedV
        self.chemoAlpha = chemoAlpha
        self.omegaMax = omegaMax
        self.omegaMin = omegaMin
        self.vOmega = vOmega
        self.cTh = cTh
        self.boundaryLength = boundaryLength
        self.cellNumInLine = cellNumInLine
        self.diffusionRateD = diffusionRateD
        self.feedRate = feedRate
        self.decayRate = decayRate
        self.consumptionRate = consumptionRate
        self.dt = dt
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.randomSeed = randomSeed
        self.overWrite = overWrite

        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        self.freqOmega = np.random.random(agentsNum) * (self.omegaMax - omegaMin) + omegaMin
        self.c = (
            # feedRate / decayRate + 
            np.random.random((cellNumInLine, cellNumInLine)) * 0.1 - 0.05
        )
        self.dx = boundaryLength / (cellNumInLine - 1)
        self._consumptionRate = consumptionRate / boundaryLength**2 * cellNumInLine**2
        self.temp = {}
        self.counts = 0
        self.temp["ocsiIdx"] = (self.positionX / self.dx).round().astype(int)

    @staticmethod
    @nb.njit
    def _direction(phaseTheta: np.ndarray) -> np.ndarray:
        direction = np.zeros((phaseTheta.shape[0], 2))
        direction[:, 0] = np.cos(phaseTheta)
        direction[:, 1] = np.sin(phaseTheta)
        return direction
    
    @property
    def dotPosition(self) -> np.ndarray:
        return self.speedV * self._direction(self.phaseTheta)
    
    @property
    def nablaC(self):
        return - np.array([ 
            (np.roll(self.c, -1, axis=0) - np.roll(self.c, 1, axis=0)),
            (np.roll(self.c, -1, axis=1) - np.roll(self.c, 1, axis=1))
        ]).transpose(1, 2, 0) / (2 * self.dx)
    
    @property
    def chemotactic(self):
        localGradC = self.nablaC[self.temp["ocsiIdx"][:, 0], self.temp["ocsiIdx"][:, 1]]
        phiC = np.arctan2(localGradC[:, 1], localGradC[:, 0])
        return self.chemoAlpha * np.linalg.norm(localGradC, axis=1) * np.sin(phiC - self.phaseTheta)

    @property
    def dotTheta(self) -> np.ndarray:
        return self.freqOmega + self.chemotactic

    @property
    def dotOmega(self) -> np.ndarray:
        localC = self.c[self.temp["ocsiIdx"][:, 0], self.temp["ocsiIdx"][:, 1]]
        return (
            self.vOmega 
            * (self.omegaMax - self.freqOmega)
            * (self.freqOmega - self.omegaMin)
            * (localC - self.cTh)
        )
    
    @property
    def nabla2C(self):
        center = -self.c
        direct_neighbors = 0.20 * (
            np.roll(self.c, 1, axis=0)
            + np.roll(self.c, -1, axis=0)
            + np.roll(self.c, 1, axis=1)
            + np.roll(self.c, -1, axis=1)
        )
        diagonal_neighbors = 0.05 * (
            np.roll(np.roll(self.c, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.c, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.c, -1, axis=0), -1, axis=1)
            + np.roll(np.roll(self.c, 1, axis=0), -1, axis=1)
        )

        out_array = center + direct_neighbors + diagonal_neighbors
        return out_array

    @staticmethod
    @nb.njit
    def _product_c(cellNumInLine: int, ocsiIdx: np.ndarray, productRateK0: np.ndarray, 
                   meaning: bool = False, spreadNum: int = 0):
        
        productC = np.zeros((cellNumInLine, cellNumInLine), dtype=np.float64)
        counts = np.zeros((cellNumInLine, cellNumInLine), dtype=np.int32)

        for i, idx in enumerate(ocsiIdx):
            for j in range(-spreadNum, 1 + spreadNum):
                for k in range(-spreadNum, 1 + spreadNum):
                    newIdx = (idx[0] + j) % cellNumInLine, (idx[1] + k) % cellNumInLine
                    productC[newIdx[0], newIdx[1]] += productRateK0[i]
                    counts[newIdx[0], newIdx[1]] += 1
        if meaning:
            counts = np.where(counts == 0, 1, counts)
            productC = productC / counts

        return productC

    @property
    def dotC(self) -> np.ndarray:
        consumption = self._product_c(cellNumInLine=self.cellNumInLine, 
                                      ocsiIdx=self.temp["ocsiIdx"], 
                                      productRateK0=np.ones(self.agentsNum, dtype=np.float64),
                                      meaning=False, spreadNum=0)
        return (
            self.diffusionRateD * self.nabla2C
            + self.feedRate
            - self.decayRate * self.c
            - self._consumptionRate * consumption
        )

    def update(self):
        self.temp["ocsiIdx"] = (self.positionX / self.dx).round().astype(int)
        dotPos = self.dotPosition
        dotTheta = self.dotTheta
        dotOmega = self.dotOmega
        dotC = self.dotC

        self.positionX = np.mod(self.positionX + dotPos * self.dt, self.boundaryLength)
        self.phaseTheta = np.mod(self.phaseTheta + dotTheta * self.dt, 2 * np.pi)
        self.freqOmega = np.clip(self.freqOmega + dotOmega * self.dt, self.omegaMin, self.omegaMax)
        self.c = np.clip(self.c + dotC * self.dt, 0, None)

        self.counts += 1

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="theta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="freqOmega", value=pd.DataFrame(self.freqOmega))
            self.store.append(key="c", value=pd.DataFrame(self.c))

    def plot(self, ax: plt.Axes = None):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        colors = np.where(self.freqOmega > 0, "#F8B08E", "#9BD5D5")
        for i in range(self.agentsNum):
            ax.add_artist(plt.Circle(
                self.positionX[i], 3 * self.boundaryLength / 200 / 2 * 0.95, zorder=1, 
                facecolor=colors[i], edgecolor="black"
            ))
        colors = np.where(self.freqOmega > 0, "#F16623", "#49B2B2")
        ax.quiver(
            self.positionX[:, 0], self.positionX[:, 1], 
            np.cos(self.phaseTheta), np.sin(self.phaseTheta),
            color=colors, width=0.004, scale=50
        )
        ax.set_xlim(0, self.boundaryLength)
        ax.set_ylim(0, self.boundaryLength)

    def __str__(self) -> str:
        return (
            f"ChemoSensingCAP(agentsNum={self.agentsNum},speedV={self.speedV:.1f},"
            f"chemoAlpha={self.chemoAlpha:.2f},"
            f"omegaMax={self.omegaMax:.1f},omegaMin={self.omegaMin:.1f},vOmega={self.vOmega:.2f},"
            f"cTh={self.cTh:.2f},boundaryLength={self.boundaryLength:.1f},"
            f"cellNumInLine={self.cellNumInLine},diffusionRateD={self.diffusionRateD:.1f},"
            f"feedRate={self.feedRate:.4f},decayRate={self.decayRate:.4f},"
            f"consumptionRate={self.consumptionRate:.1f},dt={self.dt:.2f})"
        )
    

class StateAnalysis:
    def __init__(self, model: ChemoSensingCAP = None, classDistance: float = 2, 
                 lookIndex: int = -1, showTqdm: bool = False):
        
        self.classDistance = classDistance
        self.lookIndex = lookIndex
        self.showTqdm = showTqdm
        
        if model is not None:
            self.model = model
            targetPath = f"{self.model.savePath}/{self.model}.h5"
            totalPositionX = pd.read_hdf(targetPath, key="positionX")
            totalPhaseTheta = pd.read_hdf(targetPath, key="theta")
            totalFreqOmega = pd.read_hdf(targetPath, key="freqOmega")
            totalC = pd.read_hdf(targetPath, key="c")
            
            TNum = totalPositionX.shape[0] // self.model.agentsNum
            self.TNum = TNum
            self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
            self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)
            self.totalFreqOmega = totalFreqOmega.values.reshape(TNum, self.model.agentsNum)
            self.totalC = totalC.values.reshape(TNum, model.cellNumInLine, model.cellNumInLine)
            
            self.maxC = self.totalC[-1].max()
            self.minC = self.totalC[-1].min()

    def get_state(self, index: int = -1):
        positionX = self.totalPositionX[index]
        phaseTheta = self.totalPhaseTheta[index]
        freqOmega = self.totalFreqOmega[index]
        c = self.totalC[index]

        return positionX, phaseTheta, freqOmega, c