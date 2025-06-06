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

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)


sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


class PhaseLagPatternFormation(Swarmalators2D):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0.1, deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        
        assert freqDist in ["uniform", "cauchy"]
        
        self.strengthK = strengthK
        self.distanceD0 = distanceD0
        self.phaseLagA0 = phaseLagA0
        self.boundaryLength = boundaryLength
        self.speedV = speedV
        self.freqDist = freqDist
        self.initPhaseTheta = initPhaseTheta
        self.omegaMin = omegaMin
        self.deltaOmega = deltaOmega
        self.agentsNum = agentsNum
        self.dt = dt
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.overWrite = overWrite
        
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        if initPhaseTheta is not None:
            assert len(initPhaseTheta) == agentsNum, "initPhaseTheta must match agentsNum"
            self.phaseTheta = initPhaseTheta
        if freqDist == "uniform":
            posOmega = np.random.uniform(omegaMin, omegaMin + deltaOmega, agentsNum // 2)
        else:
            posOmega = np.random.standard_cauchy(agentsNum // 2)
        self.freqOmega = np.concatenate([
            posOmega, -posOmega
        ])
        self.freqOmega = np.sort(self.freqOmega)
        self.halfBoundaryLength = boundaryLength / 2
        self.counts = 0
    
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
    def dotPhase(self) -> np.ndarray:
        return self._calc_dot_pahse(self.deltaTheta, self.A, self.freqOmega, 
                                    self.strengthK, self.phaseLagA0)

    @property
    def A(self) -> np.ndarray:
        """Adjacency matrix: 1 if |x_i - x_j| <= d0 else 0"""
        return np.where(self.distance_x(self.deltaX) <= self.distanceD0, 1, 0)

    @staticmethod
    @nb.njit
    def _calc_dot_pahse(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, 
                        K: float, phaseLagA0: float) -> np.ndarray:
        coupling = np.zeros(deltaTheta.shape[0])
        for idx in range(deltaTheta.shape[0]):
            coupling[idx] = np.mean(
                np.sin(deltaTheta[idx][A[idx] == 1] + phaseLagA0) - np.sin(phaseLagA0)
            )
        return K * coupling + omega

    def update(self):
        dotPos = self.dotPosition
        dotPhase = self.dotPhase
        
        self.positionX = np.mod(self.positionX + dotPos * self.dt, self.boundaryLength)
        self.phaseTheta = np.mod(self.phaseTheta + dotPhase * self.dt, 2 * np.pi)

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
    
    def plot(self, ax: plt.Axes = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        plt.quiver(
            self.positionX[:self.agentsNum // 2, 0], self.positionX[:self.agentsNum // 2, 1],
            np.cos(self.phaseTheta[:self.agentsNum // 2]), np.sin(self.phaseTheta[:self.agentsNum // 2]), color="red"
        )
        plt.quiver(
            self.positionX[self.agentsNum // 2:, 0], self.positionX[self.agentsNum // 2:, 1],
            np.cos(self.phaseTheta[self.agentsNum // 2:]), np.sin(self.phaseTheta[self.agentsNum // 2:]), color="#414CC7"
        )
        plt.xlim(0, self.boundaryLength)
        plt.ylim(0, self.boundaryLength)

    def __str__(self):
        return (
            f"PhaseLagPatternFormation(strengthK={self.strengthK:.3f},distanceD0={self.distanceD0:.3f},"
            f"phaseLagA0={self.phaseLagA0:.3f},boundaryLength={self.boundaryLength:.1f},"
            f"speedV={self.speedV:.1f},freqDist='{self.freqDist}',omegaMin={self.omegaMin:.3f},"
            f"deltaOmega={self.deltaOmega:.3f},agentsNum={self.agentsNum},dt={self.dt:.2f})"
        )
    

class StateAnalysis:
    def __init__(self, model: PhaseLagPatternFormation = None):
        if model is None:
            return
        self.model = model
        
        targetPath = f"{self.model.savePath}/{self.model}.h5"
        totalPositionX = pd.read_hdf(targetPath, key="positionX")
        totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
        
        TNum = totalPositionX.shape[0] // self.model.agentsNum
        self.TNum = TNum
        self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
        self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)

    def get_state(self, index: int = -1):
        positionX = self.totalPositionX[index]
        phaseTheta = self.totalPhaseTheta[index]

        return positionX, phaseTheta
    
    def plot_spatial(self, ax: plt.Axes = None, index: int = -1):
        positionX, phaseTheta = self.get_state(index)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        posIdxs = self.model.freqOmega > 0

        ax.quiver(
            positionX[posIdxs, 0], positionX[posIdxs, 1],
            np.cos(phaseTheta[posIdxs]), np.sin(phaseTheta[posIdxs]), 
            color="red", alpha=0.8
        )
        ax.quiver(
            positionX[~posIdxs, 0], positionX[~posIdxs, 1],
            np.cos(phaseTheta[~posIdxs]), np.sin(phaseTheta[~posIdxs]), color="#414CC7"
        )
        ax.set_xlim(0, self.model.boundaryLength)
        ax.set_ylim(0, self.model.boundaryLength)