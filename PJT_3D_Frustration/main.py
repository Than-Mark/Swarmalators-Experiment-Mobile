import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from scipy.spatial import Delaunay
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import json
import sys
import os
import shutil

randomSeed = 10

if "ipykernel_launcher.py" in sys.argv[0]:
    # from tqdm.notebook import tqdm
    from tqdm import tqdm
else:
    from tqdm import tqdm

colors = ["#403990", "#3A76D6", "#FFC001", "#F46F43", "#FF0000"]
cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors)

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
)
with open("../swarmalatorlib/hex_colors.json", "r", encoding="utf-8") as f:
    hexColors = json.load(f)
hexCmap = mcolors.LinearSegmentedColormap.from_list("cmap", hexColors)

sys.path.append("..")
from swarmalatorlib.template import Swarmalators


class Frustration3D(Swarmalators):
    def __init__(self, strengthA: float, strengthB: float, 
                 distanceD0: float, phaseLagAlpha0: float,
                 boundaryLength: float = 10, speedV: float = 3.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:

        self.strengthA = strengthA
        self.strengthB = strengthB
        self.distanceD0 = distanceD0
        self.phaseLagAlpha0 = phaseLagAlpha0
        self.boundaryLength = boundaryLength
        self.speedV = speedV
        self.matrixK = np.array([
            [ strengthA * np.cos(phaseLagAlpha0), strengthA * np.sin(phaseLagAlpha0), 0],
            [-strengthA * np.sin(phaseLagAlpha0), strengthA * np.cos(phaseLagAlpha0), 0],
            [0, 0, strengthB]
        ])

        self.agentsNum = agentsNum
        self.dt = dt
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.randomSeed = randomSeed
        self.overWrite = overWrite
        
        np.random.seed(randomSeed)
        self.positionX = np.ascontiguousarray(np.random.random((agentsNum, 3)) * boundaryLength)
        phi = np.random.random(agentsNum) * 2 * np.pi
        xi = np.random.random(agentsNum) * 2 - 1
        cosTheta = xi
        sinTheta = np.sqrt(1 - cosTheta**2)
        self.phaseSigma = np.ascontiguousarray(np.array([
            np.cos(phi) * sinTheta,
            np.sin(phi) * sinTheta,
            cosTheta
        ]).T)

        self.halfBoundaryLength = boundaryLength / 2
        self.counts = 0
        self.dotPhaseParams = (
            self.boundaryLength,
            self.halfBoundaryLength,
            self.distanceD0,
            self.matrixK,
            self.phaseLagAlpha0,
        )
        self.stateAnalysisClass = StateAnalysis3D

    @staticmethod
    @nb.njit
    def _calc_dot_phase(positionX: np.ndarray, phaseSigma: np.ndarray, 
                        params: Tuple[float]) -> np.ndarray:
        agentsNum = positionX.shape[0]
        boundaryLength, halfBoundaryLength, distanceD0, matrixK, phaseLagAlpha0 = params

        coupling = np.zeros((agentsNum, 3))
        for i in range(agentsNum):
            xDiff = np.abs(positionX[:, 0] - positionX[i, 0])
            yDiff = np.abs(positionX[:, 1] - positionX[i, 1])
            zDiff = np.abs(positionX[:, 2] - positionX[i, 2])
            neighborIdxs = np.where(
                (xDiff < distanceD0) | (boundaryLength - xDiff < distanceD0) & 
                (yDiff < distanceD0) | (boundaryLength - yDiff < distanceD0) &
                (zDiff < distanceD0) | (boundaryLength - zDiff < distanceD0)
            )[0]
            if neighborIdxs.size == 0:
                continue

            subX = positionX[i] - positionX[neighborIdxs]
            deltaX = positionX[i] - (
                positionX[neighborIdxs] * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) + 
                (positionX[neighborIdxs] - boundaryLength) * (subX < -halfBoundaryLength) + 
                (positionX[neighborIdxs] + boundaryLength) * (subX > halfBoundaryLength)
            )
            distance = np.sqrt(np.sum(deltaX**2, axis=1))
            A = np.where(distance <= distanceD0)[0]
            if A.size == 0:
                continue

            part1 = matrixK @ phaseSigma[neighborIdxs[A]].sum(axis=0)
            part2 = (phaseSigma[i] @ matrixK @ phaseSigma[neighborIdxs[A]].T).sum(axis=0) * phaseSigma[i]
            coupling[i] = (part1 - part2) / A.size
        return coupling
    
    @property
    def dotPhase(self) -> np.ndarray:
        return self._calc_dot_phase(self.positionX, self.phaseSigma, self.dotPhaseParams)
    
    @property
    def dotPosition(self) -> np.ndarray:
        return self.speedV * self.phaseSigma
    
    @staticmethod
    @nb.njit
    def _update(positionX: np.ndarray, phaseSigma: np.ndarray,
                dotPos: np.ndarray, dotPhase: np.ndarray, dt: float,
                boundaryLength: float) -> Tuple[np.ndarray, np.ndarray]:
        positionX = (positionX + dotPos * dt) % boundaryLength
        phaseSigma = phaseSigma + dotPhase * dt
        norm = np.sqrt((phaseSigma**2).sum(axis=1)).reshape(-1, 1)
        phaseSigma = phaseSigma / norm
        return positionX, phaseSigma

    def update(self):
        dotPos = self.dotPosition
        dotPhase = self.dotPhase

        self.positionX, self.phaseSigma = self._update(
            self.positionX, self.phaseSigma, dotPos, dotPhase, self.dt, self.boundaryLength
        )

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseSigma", value=pd.DataFrame(self.phaseSigma))

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"strengthA={self.strengthA:.3f},strengthB={self.strengthB:.3f},"
            f"distanceD0={self.distanceD0:.3f},phaseLagAlpha0={self.phaseLagAlpha0:.3f},"
            f"boundaryLength={self.boundaryLength:.1f},speedV={self.speedV:.1f},"
            f"agentsNum={self.agentsNum},dt={self.dt:.3f},"
            f"shotsnaps={self.shotsnaps},randomSeed={self.randomSeed}"
            ")"
        )
    

class StateAnalysis3D:
    def __init__(self, model: Frustration3D = None):
        if model is None:
            return
        self.model = model
        
        targetPath = f"{self.model.savePath}/{self.model}.h5"
        
        totalPhaseSigma = pd.read_hdf(targetPath, key="phaseSigma")
        TNum = totalPhaseSigma.shape[0] // self.model.agentsNum
        self.TNum = TNum
        self.totalPhaseSigma = totalPhaseSigma.values.reshape(TNum, self.model.agentsNum, 3)

        totalPositionX = pd.read_hdf(targetPath, key="positionX")
        self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 3)

    def get_state(self, index: int = -1):
        positionX = self.totalPositionX[index]
        phaseSigma = self.totalPhaseSigma[index]

        return positionX, phaseSigma