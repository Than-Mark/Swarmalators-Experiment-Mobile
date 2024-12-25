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

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


class PatternFormation(Swarmalators2D):
    def __init__(self, strengthLambda: float, alpha: float, boundaryLength: float = 10, 
                 productRateK0: float = 1, decayRateKd: float = 1, c0: float = 5, 
                 chemotacticStrengthBetaR: float = 1, diffusionRateDc: float = 1, 
                 epsilon: float = 10, cellNumInLine: int = 50, 
                 typeA: str = "distanceWgt", agentsNum: int=1000, dt: float=0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10, 
                 distribution: str = "uniform", randomSeed: int = 10, overWrite: bool = False) -> None:
        assert distribution in ["uniform"]
        assert typeA in ["distanceWgt"]

        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.c = np.random.rand(cellNumInLine, cellNumInLine)
        self.cellNumInLine = cellNumInLine
        self.cPosition = np.array(list(product(np.linspace(0, boundaryLength, cellNumInLine), repeat=2)))
        self.dx = boundaryLength / (cellNumInLine - 1)
        self.agentsNum = agentsNum
        self.productRateK0 = productRateK0
        self.decayRateKd = decayRateKd
        self.diffusionRateDc = diffusionRateDc
        self.chemotacticStrengthBetaR = chemotacticStrengthBetaR
        self.c0 = c0
        self.epsilon = epsilon
        self.dt = dt
        self.speedV = 3
        self.alpha = alpha
        if distribution == "uniform":
            self.freqOmega = np.concatenate([
                np.random.uniform(1, 3, size=agentsNum // 2),
                np.random.uniform(-3, -1, size=agentsNum // 2)
            ])
        elif distribution == "normal":
            self.freqOmega = np.concatenate([
                np.random.normal(loc=3, scale=0.5, size=agentsNum // 2),
                np.random.normal(loc=-3, scale=0.5, size=agentsNum // 2)
            ])

        self.typeA = typeA
        self.distribution = distribution
        self.strengthLambda = strengthLambda
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.boundaryLength = boundaryLength
        self.halfBoundaryLength = boundaryLength / 2
        self.randomSeed = randomSeed
        self.overWrite = overWrite

        self.temp = dict()
        # The order of variable definitions has a dependency relationship
        self.temp["direction"] = self._direction(self.phaseTheta)
        self.temp["CXDistanceWgtA"] = self._distance_wgt_A(self.distance_x(self.deltaCX), self.alpha)
        self.temp["dotTheta"] = self.dotTheta
        self.temp["dotC"] = self.dotC

    def plot(self, ax: plt.Axes = None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            self.positionX[:self.agentsNum // 2, 0], self.positionX[:self.agentsNum // 2, 1],
            color="#F8B08E", s=10  # edgecolors="black"
        )
        ax.scatter(
            self.positionX[self.agentsNum // 2:, 0], self.positionX[self.agentsNum // 2:, 1],
            color="#9BD5D5", s=10  # edgecolors="black"
        )
        ax.quiver(
            self.positionX[:self.agentsNum // 2, 0], self.positionX[:self.agentsNum // 2, 1],
            np.cos(self.phaseTheta[:self.agentsNum // 2]), np.sin(self.phaseTheta[:self.agentsNum // 2]), color="#F16623"
        )
        ax.quiver(
            self.positionX[self.agentsNum // 2:, 0], self.positionX[self.agentsNum // 2:, 1],
            np.cos(self.phaseTheta[self.agentsNum // 2:]), np.sin(self.phaseTheta[self.agentsNum // 2:]), color="#49B2B2"
        )
        ax.set_xlim(0, self.boundaryLength)
        ax.set_ylim(0, self.boundaryLength)

    def plot_field(self, ax: plt.Axes = None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.contourf(self.c, cmap='viridis', levels=50)

    @staticmethod
    @nb.njit
    def _distance_wgt_A(distance: np.ndarray, alpha: float):
        return np.exp(-distance / alpha)

    @property
    def A(self):
        if self.typeA == "heaviside":
            return self.distance_x(self.deltaX) <= self.alpha
        elif self.typeA == "distanceWgt":
            return self._distance_wgt_A(self.distance_x(self.deltaX), self.alpha)

    @staticmethod
    @nb.njit
    def _distance_wgt_product_c(distance_wgt_A: np.ndarray, productRateK0: float):
        return distance_wgt_A.sum(axis=0) / distance_wgt_A.shape[0] * productRateK0

    @property
    def productC(self):
        if self.typeA == "heaviside":
            value = (self.distance_x(self.deltaCX) <= self.alpha).mean(axis=0) * self.productRateK0
        elif self.typeA == "distanceWgt":
            value = self._distance_wgt_product_c(
                self._distance_wgt_A(self.temp["CXDistanceWgtA"], self.alpha), 
                self.productRateK0
            )
        return self._reshape_product_c(value, self.cellNumInLine)

    @staticmethod
    @nb.njit
    def _reshape_product_c(cPosition: np.ndarray, cellNumInLine: int):
        return np.reshape(cPosition, (cellNumInLine, cellNumInLine))

    @property
    def decayC(self):
        return self.c * self.decayRateKd
    
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
        return out_array / (self.dx ** 2)
    
    @property
    def nablaC(self):
        return np.array([
            (np.roll(self.c, 1, axis=1) - np.roll(self.c, -1, axis=1)) / (2 * self.dx), 
            (np.roll(self.c, 1, axis=0) - np.roll(self.c, -1, axis=0)) / (2 * self.dx)
        ]).transpose(1, 2, 0)

    @property
    def diffusionC(self):
        return self.diffusionRateDc * self.nabla2C

    @property
    def growthLimitC(self):
        return self.epsilon * (self.c0 - self.c) ** 3        

    @property
    def deltaCX(self):
        return self._delta_x(self.cPosition, self.positionX[:, np.newaxis], 
                             self.boundaryLength, self.halfBoundaryLength)

    @property
    def deltaX(self) -> np.ndarray:
        return self._delta_x(self.positionX, self.positionX[:, np.newaxis], 
                             self.boundaryLength, self.halfBoundaryLength)

    @staticmethod
    @nb.njit
    def _delta_x(positionX: np.ndarray, others: np.ndarray,
                 boundaryLength: float, halfBoundaryLength: float) -> np.ndarray:
        subX = positionX - others
        return (
            subX * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) +
            (subX + boundaryLength) * (subX < -halfBoundaryLength) +
            (subX - boundaryLength) * (subX > halfBoundaryLength)
        )

    @property
    def chemotactic(self):
        idxs = (self.positionX / self.dx).round().astype(int)
        localGradC = self.nablaC[idxs[:, 0], idxs[:, 1]]
        return self.chemotacticStrengthBetaR * (
            self.temp["direction"][:, 0] * localGradC[:, 1] -
            self.temp["direction"][:, 1] * localGradC[:, 0]
        )

    @property
    def dotTheta(self):
        return self._dotTheta(self.phaseTheta, self.freqOmega, self.chemotactic, 
                              self.strengthLambda, self.A)

    @staticmethod
    @nb.njit
    def _dotTheta(phaseTheta: np.ndarray, freqOmega: np.ndarray, 
                  chemotactic: np.ndarray, strengthLambda: float, 
                  A: np.ndarray):
        adjMatrixTheta = (
            np.repeat(phaseTheta, phaseTheta.shape[0])
            .reshape(phaseTheta.shape[0], phaseTheta.shape[0])
        )
        return freqOmega + chemotactic + strengthLambda * np.sum(A * np.sin(
            adjMatrixTheta - phaseTheta
        ), axis=0)
        
    @property
    def dotC(self):
        return self.productC - self.decayC + self.diffusionC + self.growthLimitC

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="dotTheta", value=pd.DataFrame(self.temp["dotTheta"]))
            self.store.append(key="c", value=pd.DataFrame(self.c))
            self.store.append(key="dotC", value=pd.DataFrame(self.temp["dotC"]))

    @staticmethod
    @nb.njit
    def _direction(phaseTheta: np.ndarray) -> np.ndarray:
        direction = np.zeros((phaseTheta.shape[0], 2))
        direction[:, 0] = np.cos(phaseTheta)
        direction[:, 1] = np.sin(phaseTheta)
        return direction

    def update(self):
        # The order of variable definitions has a dependency relationship
        self.temp["CXDistanceWgtA"] = self._distance_wgt_A(self.distance_x(self.deltaCX), self.alpha)
        self.temp["dotTheta"] = self.dotTheta
        self.temp["dotC"] = self.dotC
        self.temp["direction"] = self._direction(self.phaseTheta)
        self.positionX += self.speedV * self.temp["direction"] * self.dt
        self.positionX = np.mod(self.positionX, self.boundaryLength)
        self.phaseTheta += self.temp["dotTheta"] * self.dt
        self.c += self.temp["dotC"] * self.dt
        self.c[self.c < 0] = 0
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi

    def __str__(self) -> str:
        
        name =  (
            f"PF_K{self.strengthLambda:.3f}_a{self.alpha:.2f}"
            f"_b{self.chemotacticStrengthBetaR:.1f}"
            f"_r{self.randomSeed}"
        )
        
        return name

    def close(self):
        if self.store is not None:
            self.store.close()


class GSPatternFormation(PatternFormation):
    def __init__(self, strengthLambda: float, alpha: float, boundaryLength: float = 10, 
                 productRateUK0: float = 1, productRateVK0: float = 1,
                 decayRateKd: float = 1, speedV: float = 3,
                 chemoBetaU: float = 1, chemoBetaV: float = 1, 
                 diffusionRateDu: float = 1, diffusionRateDv: float = 1, 
                 cellNumInLine: int = 50, 
                 typeA: str = "distanceWgt", agentsNum: int=1000, dt: float=0.01, 
                 distribution: str = "uniform", omegaMean: float = 2, omegaStd: float = 1,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10, 
                 randomSeed: int = 10, overWrite: bool = False) -> None:

        assert distribution in ["uniform", "normal"]
        assert typeA in ["distanceWgt", "heaviside"]

        self.halfAgentsNum = agentsNum // 2

        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.cellNumInLine = cellNumInLine
        self.cPosition = np.array(list(product(np.linspace(0, boundaryLength, cellNumInLine), repeat=2)))
        self.dx = boundaryLength / (cellNumInLine - 1)
        self.agentsNum = agentsNum
        self.productRateUK0 = productRateUK0
        self.productRateVK0 = productRateVK0
        self.decayRateKd = decayRateKd
        self.diffusionRateDu = diffusionRateDu
        self.diffusionRateDv = diffusionRateDv
        self.dt = dt
        self.speedV = speedV
        self.alpha = alpha
        if distribution == "uniform":
            freqOmega = np.random.uniform(omegaMean - omegaStd, omegaMean + omegaStd, size=self.halfAgentsNum)
        elif distribution == "normal":
            freqOmega = np.random.normal(loc=omegaMean, scale=omegaStd, size=self.halfAgentsNum)
        self.freqOmega = np.concatenate([freqOmega, -freqOmega])
        self.omegaMean = omegaMean
        self.omegaStd = omegaStd

        self.typeA = typeA
        self.distribution = distribution
        self.strengthLambda = strengthLambda
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.boundaryLength = boundaryLength
        self.halfBoundaryLength = boundaryLength / 2
        self.randomSeed = randomSeed
        self.overWrite = overWrite

        self.u = np.random.rand(cellNumInLine, cellNumInLine)
        self.v = np.random.rand(cellNumInLine, cellNumInLine)
        self.chemoBetaUArray = chemoBetaU * np.concatenate([
            np.zeros(self.halfAgentsNum), np.ones(self.halfAgentsNum)
        ])
        self.chemoBetaVArray = chemoBetaV * np.concatenate([
            np.ones(self.halfAgentsNum), np.zeros(self.halfAgentsNum)
        ])
        self.chemoBetaU = chemoBetaU
        self.chemoBetaV = chemoBetaV

        self.temp = dict()
        # The order of variable definitions has a dependency relationship
        self.temp["direction"] = self._direction(self.phaseTheta)
        self.temp["ocsiIdx"] = (self.positionX / self.dx).round().astype(int)
        self.temp["dotTheta"] = self.dotTheta
        self.temp["dotU"] = self.dotU
        self.temp["dotV"] = self.dotV
    
    @property
    def nablaU(self):
        return np.array([
            (np.roll(self.u, 1, axis=1) - np.roll(self.u, -1, axis=1)) / (2 * self.dx), 
            (np.roll(self.u, 1, axis=0) - np.roll(self.u, -1, axis=0)) / (2 * self.dx)
        ]).transpose(1, 2, 0)
    
    @property
    def nablaV(self):
        return np.array([
            (np.roll(self.v, 1, axis=1) - np.roll(self.v, -1, axis=1)) / (2 * self.dx), 
            (np.roll(self.v, 1, axis=0) - np.roll(self.v, -1, axis=0)) / (2 * self.dx)
        ]).transpose(1, 2, 0)

    @staticmethod
    @nb.njit
    def _product_c(cellNumInLine: int, ocsiIdx: np.ndarray, productRateK0: float):
        productC = np.zeros((cellNumInLine, cellNumInLine), dtype=np.float64)
        for idx in ocsiIdx:
            productC[idx[0], idx[1]] = productC[idx[0], idx[1]] + 1
        return productC * productRateK0

    @property
    def productU(self):
        return self._product_c(
            self.cellNumInLine, 
            self.temp["ocsiIdx"][:self.halfAgentsNum],
            self.productRateUK0
        )
    
    @property
    def productV(self):
        return self._product_c(
            self.cellNumInLine, 
            self.temp["ocsiIdx"][self.halfAgentsNum:],
            self.productRateVK0
        )

    @property
    def chemotactic(self):
        localGradU = self.nablaU[self.temp["ocsiIdx"][:, 0], self.temp["ocsiIdx"][:, 1]]
        localGradV = self.nablaV[self.temp["ocsiIdx"][:, 0], self.temp["ocsiIdx"][:, 1]]
        return self.chemoBetaUArray * (
            self.temp["direction"][:, 0] * localGradU[:, 1] - 
            self.temp["direction"][:, 1] * localGradU[:, 0]
        ) + self.chemoBetaVArray * (
            self.temp["direction"][:, 0] * localGradV[:, 1] -
            self.temp["direction"][:, 1] * localGradV[:, 0]
        )
    
    @property
    def nabla2U(self):
        center = -self.u
        direct_neighbors = 0.20 * (
            np.roll(self.u, 1, axis=0)
            + np.roll(self.u, -1, axis=0)
            + np.roll(self.u, 1, axis=1)
            + np.roll(self.u, -1, axis=1)
        )
        diagonal_neighbors = 0.05 * (
            np.roll(np.roll(self.u, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.u, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.u, -1, axis=0), -1, axis=1)
            + np.roll(np.roll(self.u, 1, axis=0), -1, axis=1)
        )

        out_array = center + direct_neighbors + diagonal_neighbors
        return out_array / (self.dx ** 2)
    
    @property
    def nabla2V(self):
        center = -self.v
        direct_neighbors = 0.20 * (
            np.roll(self.v, 1, axis=0)
            + np.roll(self.v, -1, axis=0)
            + np.roll(self.v, 1, axis=1)
            + np.roll(self.v, -1, axis=1)
        )
        diagonal_neighbors = 0.05 * (
            np.roll(np.roll(self.v, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.v, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(self.v, -1, axis=0), -1, axis=1)
            + np.roll(np.roll(self.v, 1, axis=0), -1, axis=1)
        )

        out_array = center + direct_neighbors + diagonal_neighbors
        return out_array / (self.dx ** 2)
    
    @property
    def diffusionU(self):
        return self.diffusionRateDu * self.nabla2U
    
    @property
    def diffusionV(self):
        return self.diffusionRateDv * self.nabla2V
    
    @property
    def dotU(self):
        return (
            self.productU 
            - self.u * self.decayRateKd 
            + self.diffusionU
        )
    
    @property
    def dotV(self):
        return (
            self.productV  
            - self.v * self.decayRateKd 
            + self.diffusionV
        )

    def update(self):
        self.temp["ocsiIdx"] = (self.positionX / self.dx).round().astype(int)
        self.temp["dotTheta"] = self.dotTheta
        self.temp["dotU"] = self.dotU
        self.temp["dotV"] = self.dotV
        self.temp["direction"] = self._direction(self.phaseTheta)
        self.positionX = np.mod(
            self.positionX + self.speedV * self.temp["direction"] * self.dt, 
            self.boundaryLength
        )
        self.phaseTheta += self.temp["dotTheta"] * self.dt
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi
        self.u += self.temp["dotU"] * self.dt
        self.v += self.temp["dotV"] * self.dt
        self.u[self.u < 0] = 0
        self.v[self.v < 0] = 0

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="dotTheta", value=pd.DataFrame(self.temp["dotTheta"]))
            self.store.append(key="u", value=pd.DataFrame(self.u))
            self.store.append(key="v", value=pd.DataFrame(self.v))

    def __str__(self) -> str:
            
        name =  (
            f"GSPF_K{self.strengthLambda:.3f}_a{self.alpha:.2f}"
            f"_bu{self.chemoBetaU:.1f}_bv{self.chemoBetaV:.1f}"
            f"_pu{self.productRateUK0:.2f}_pv{self.productRateVK0:.2f}"
            f"_Kd{self.decayRateKd:.2f}_sV{self.speedV:.1f}"
            f"_Du{self.diffusionRateDu:.3f}_Dv{self.diffusionRateDv:.3f}"
            f"_r{self.randomSeed}"
            f"_oD({self.distribution},{self.omegaMean:.1f},{self.omegaStd:.1f})"
        )
        
        return name
    

class ChemotacticLotkaVolterra(PatternFormation):
    def __init__(self, k1: float, k2: float, k3: float, k4: float,
                 boundaryLength: float = 10, speedV: float = 3, 
                 diameter: float = 0.1, repelPower: float = 1,
                 chemoAlpha1: float = 1, chemoAlpha2: float = 1,
                 diffusionRateD1: float = 1, diffusionRateD2: float = 1,
                 cellNumInLine: int = 50, agentsNum: int=1000, dt: float=0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        
        np.random.seed(randomSeed)

        self.randomSeed = randomSeed
        self.cellNumInLine = cellNumInLine
        self.cPosition = np.array(list(product(np.linspace(0, boundaryLength, cellNumInLine), repeat=2)))
        self.dx = boundaryLength / (cellNumInLine - 1)
        self.boundaryLength = boundaryLength
        self.halfBoundaryLength = boundaryLength / 2
        self.agentsNum = agentsNum
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.diffusionRateD1 = diffusionRateD1
        self.diffusionRateD2 = diffusionRateD2
        self.dt = dt
        self.speedV = speedV
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.overWrite = overWrite
        self.halfAgentsNum = agentsNum // 2
        self.chemoAlpha1Mat = chemoAlpha1 * np.concatenate([
            np.zeros(self.halfAgentsNum), np.ones(self.halfAgentsNum)
        ])
        self.chemoAlpha2Mat = chemoAlpha2 * np.concatenate([
            np.ones(self.halfAgentsNum), np.zeros(self.halfAgentsNum)
        ])
        self.chemoAlpha1 = chemoAlpha1
        self.chemoAlpha2 = chemoAlpha2
        self.diameter = diameter
        self.repelPower = repelPower

        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        # self.positionX = np.concatenate([
        #     np.random.random((self.halfAgentsNum, 2)) * boundaryLength / 2,
        #     np.random.random((self.halfAgentsNum, 2)) * boundaryLength / 2 + boundaryLength / 2
        # ])
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.c1 = np.random.rand(cellNumInLine, cellNumInLine)
        self.c2 = np.random.rand(cellNumInLine, cellNumInLine)

        self.temp = dict()
        # The order of variable definitions has a dependency relationship
        self.temp["direction"] = self._direction(self.phaseTheta)
        self.temp["ocsiIdx"] = (self.positionX / self.dx).round().astype(int)
        self.temp["dotTheta"] = self.dotTheta
        self.temp["dotC1"] = self.dotC1
        self.temp["dotC2"] = self.dotC2

    def plot(self, ax: plt.Axes = None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            self.positionX[:self.agentsNum // 2, 0], self.positionX[:self.agentsNum // 2, 1],
            color="#F8B08E", s=100 * self.diameter / 0.1 * (2.5 / self.boundaryLength) # edgecolors="black"
        )
        ax.scatter(
            self.positionX[self.agentsNum // 2:, 0], self.positionX[self.agentsNum // 2:, 1],
            color="#9BD5D5", s=100 * self.diameter / 0.1 * (2.5 / self.boundaryLength)  # edgecolors="black"
        )
        ax.quiver(
            self.positionX[:self.agentsNum // 2, 0], self.positionX[:self.agentsNum // 2, 1],
            np.cos(self.phaseTheta[:self.agentsNum // 2]), np.sin(self.phaseTheta[:self.agentsNum // 2]), color="#F16623"
        )
        ax.quiver(
            self.positionX[self.agentsNum // 2:, 0], self.positionX[self.agentsNum // 2:, 1],
            np.cos(self.phaseTheta[self.agentsNum // 2:]), np.sin(self.phaseTheta[self.agentsNum // 2:]), color="#49B2B2"
        )
        ax.set_xlim(0, self.boundaryLength)
        ax.set_ylim(0, self.boundaryLength)

    @staticmethod
    @nb.njit
    def _calc_repulsion(divDeltaX_2: np.ndarray, distance: float, diameter: float):
        return np.sum(divDeltaX_2 * (distance < diameter), axis=1)

    @property
    def shortRepulsion(self):
        self.temp["deltaX"] = self.deltaX
        self.temp["distanceX2"] = self.distance_x_2(self.temp["deltaX"])
        return -self._calc_repulsion(
            divDeltaX_2=self.div_distance_power(self.temp["deltaX"], power=self.repelPower), 
            distance=self.temp["distanceX2"], 
            diameter=self.diameter
        )
        # return (
        #     self.div_distance_power(numerator=self.deltaX, power=2) * 
        #     (self.temp["distanceX2"] < self.radius)
        # ).sum(axis=1)

    @property
    def nablaC1(self):
        return np.array([
            (np.roll(self.c1, 1, axis=1) - np.roll(self.c1, -1, axis=1)) / (2 * self.dx), 
            (np.roll(self.c1, 1, axis=0) - np.roll(self.c1, -1, axis=0)) / (2 * self.dx)
        ]).transpose(1, 2, 0)
    
    @property
    def nablaC2(self):
        return np.array([
            (np.roll(self.c2, 1, axis=1) - np.roll(self.c2, -1, axis=1)) / (2 * self.dx), 
            (np.roll(self.c2, 1, axis=0) - np.roll(self.c2, -1, axis=0)) / (2 * self.dx)
        ]).transpose(1, 2, 0)

    @staticmethod
    @nb.njit
    def _product_c(cellNumInLine: int, ocsiIdx: np.ndarray, productRateK0: np.ndarray):
        sumCounts = np.zeros((cellNumInLine, cellNumInLine), dtype=np.float64)
        for idx in ocsiIdx:
            sumCounts[idx[0], idx[1]] = sumCounts[idx[0], idx[1]] + 1
        return sumCounts * productRateK0
    
    @property
    def productC1(self):
        return self._product_c(
            self.cellNumInLine, 
            self.temp["ocsiIdx"][:self.halfAgentsNum],
            self.c1 * (self.k1 - self.k2 * self.c2)
        )
        # return self.c1 * (self.k1 - self.k2 * self.c2)
    
    @property
    def productC2(self):
        return self._product_c(
            self.cellNumInLine, 
            self.temp["ocsiIdx"][self.halfAgentsNum:],
            self.c2 * (self.k3 * self.c1 - self.k4)
        )
        # return self.c2 * (self.k3 * self.c1 - self.k4)

    @property
    def chemotactic(self):
        localGradC1 = self.nablaC1[self.temp["ocsiIdx"][:, 0], self.temp["ocsiIdx"][:, 1]]
        localGradC2 = self.nablaC2[self.temp["ocsiIdx"][:, 0], self.temp["ocsiIdx"][:, 1]]
        return self.chemoAlpha1Mat * (
            self.temp["direction"][:, 0] * localGradC1[:, 1] - 
            self.temp["direction"][:, 1] * localGradC1[:, 0]
        ) + self.chemoAlpha2Mat * (
            self.temp["direction"][:, 0] * localGradC2[:, 1] -
            self.temp["direction"][:, 1] * localGradC2[:, 0]
        )

    def _nabla2(self, c):
        center = -c
        direct_neighbors = 0.20 * (
            np.roll(c, 1, axis=0)
            + np.roll(c, -1, axis=0)
            + np.roll(c, 1, axis=1)
            + np.roll(c, -1, axis=1)
        )
        diagonal_neighbors = 0.05 * (
            np.roll(np.roll(c, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(c, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(c, -1, axis=0), -1, axis=1)
            + np.roll(np.roll(c, 1, axis=0), -1, axis=1)
        )

        out_array = center + direct_neighbors + diagonal_neighbors
        return out_array / (self.dx ** 2)
    
    @property
    def diffusionC1(self):
        return self._nabla2(self.c1) * self.diffusionRateD1
    
    @property
    def diffusionC2(self):
        return self._nabla2(self.c2) * self.diffusionRateD2
    
    @property
    def dotC1(self):
        return self.productC1 + self.diffusionC1
    
    @property
    def dotC2(self):
        return self.productC2 + self.diffusionC2
    
    @property
    def dotTheta(self):
        return self.chemotactic
        # return self._dotTheta(self.phaseTheta, self.chemotactic, 
        #                       0, 0)

    @staticmethod
    @nb.njit
    def _dotTheta(phaseTheta: np.ndarray, 
                  chemotactic: np.ndarray, strengthLambda: float, 
                  A: np.ndarray):
        # adjMatrixTheta = (
        #     np.repeat(phaseTheta, phaseTheta.shape[0])
        #     .reshape(phaseTheta.shape[0], phaseTheta.shape[0])
        # )
        return chemotactic
        # + strengthLambda * np.sum(A * np.sin(
        #     adjMatrixTheta - phaseTheta
        # ), axis=0)

    def update(self):
        self.temp["ocsiIdx"] = (self.positionX / self.dx).round().astype(int)
        self.temp["dotTheta"] = self.dotTheta
        self.temp["dotC1"] = self.dotC1
        self.temp["dotC2"] = self.dotC2
        self.temp["direction"] = self._direction(self.phaseTheta)
        self.positionX = np.mod(
            self.positionX + (self.speedV * self.temp["direction"] + self.shortRepulsion) * self.dt, 
            self.boundaryLength
        )
        self.phaseTheta += self.temp["dotTheta"] * self.dt
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi
        self.c1 += self.temp["dotC1"] * self.dt
        self.c2 += self.temp["dotC2"] * self.dt
        self.c1[self.c1 < 0] = 0
        self.c2[self.c2 < 0] = 0
    
    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="dotTheta", value=pd.DataFrame(self.temp["dotTheta"]))
            self.store.append(key="c1", value=pd.DataFrame(self.c1))
            self.store.append(key="c2", value=pd.DataFrame(self.c2))

    def __str__(self) -> str:
                
        name =  (
            f"CLV_K1{self.k1:.3f}_K2{self.k2:.3f}_K3{self.k3:.3f}_K4{self.k4:.3f}"
            f"_a1{self.chemoAlpha1:.1f}_a2{self.chemoAlpha2:.1f}"
            f"_D1{self.diffusionRateD1:.3f}_D2{self.diffusionRateD2:.3f}"
            f"_sV{self.speedV:.1f}_d{self.diameter:.1f}_rP{self.repelPower:.1f}"
            f"_bL{self.boundaryLength:.1f}_dt{self.dt:.2f}_cN{self.cellNumInLine}"
            f"_r{self.randomSeed}"
        )
        
        return name


class StateAnalysis:
    def __init__(self, model: ChemotacticLotkaVolterra = None, classDistance: float = 2, 
                 lookIndex: int = -1, showTqdm: bool = False):
        
        self.classDistance = classDistance
        self.lookIndex = lookIndex
        self.showTqdm = showTqdm
        
        if model is not None:
            self.model = model
            targetPath = f"{self.model.savePath}/{self.model}.h5"
            totalPositionX = pd.read_hdf(targetPath, key="positionX")
            totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
            totalDotTheta = pd.read_hdf(targetPath, key="dotTheta")
            totalC1 = pd.read_hdf(targetPath, key="c1")
            totalC2 = pd.read_hdf(targetPath, key="c2")
            
            TNum = totalPositionX.shape[0] // self.model.agentsNum
            self.TNum = TNum
            self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
            self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)
            self.totalDotTheta = totalDotTheta.values.reshape(TNum, self.model.agentsNum)
            self.totalC1 = totalC1.values.reshape(TNum, model.cellNumInLine, model.cellNumInLine)
            self.totalC2 = totalC2.values.reshape(TNum, model.cellNumInLine, model.cellNumInLine)
            self.maxC1 = totalC1.values.max()
            self.maxC2 = totalC2.values.max()

            if self.showTqdm:
                self.iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
            else:
                self.iterObject = range(1, self.totalPhaseTheta.shape[0])

    def get_state(self, index: int = -1):
        positionX = self.totalPositionX[index]
        phaseTheta = self.totalPhaseTheta[index]
        dotTheta = self.totalDotTheta[index]

        return positionX, phaseTheta, dotTheta

    @staticmethod
    @nb.njit
    def _calc_centers(positionX, phaseTheta, dotTheta, speedV, dt):
        centers = np.zeros((positionX.shape[0], 2))
        centers[:, 0] = positionX[:, 0] - speedV * dt / dotTheta * np.sin(phaseTheta)
        centers[:, 1] = positionX[:, 1] + speedV * dt / dotTheta * np.cos(phaseTheta)

        return centers

    @property
    def centers(self):
        
        lastPositionX, lastPhaseTheta, lastDotTheta = self.get_state(self.lookIndex)
        
        centers = self._calc_centers(
            lastPositionX, lastPhaseTheta, lastDotTheta, self.model.speedV, self.model.dt
        )

        return np.mod(centers, self.model.boundaryLength)

    @property
    def centersNoMod(self):
            
        lastPositionX, lastPhaseTheta, lastDotTheta = self.get_state(self.lookIndex)
        
        centers = self._calc_centers(
            lastPositionX, lastPhaseTheta, lastDotTheta, self.model.speedV, self.model.dt
        )

        return centers
         
    
    def adj_distance(self, positionX, others):
        return self._adj_distance(
            positionX, others, self.model.boundaryLength, self.model.halfBoundaryLength
        )

    @staticmethod
    @nb.njit
    def _adj_distance(positionX, others, boundaryLength, halfLength):
        subX = positionX - others
        adjustOthers = (
            others * (-halfLength <= subX) * (subX <= halfLength) + 
            (others - boundaryLength) * (subX < -halfLength) + 
            (others + boundaryLength) * (subX > halfLength)
        )
        adjustSubX = positionX - adjustOthers
        return np.sqrt(np.sum(adjustSubX ** 2, axis=-1))
    
    @staticmethod
    @nb.njit
    def _calc_classes(centers, classDistance, totalDistances):
        classes = [[0]]
        classNum = 1
        nonClassifiedOsci = np.arange(1, centers.shape[0])

        for i in nonClassifiedOsci:
            newClass = True

            for classI in range(len(classes)):
                distance = classDistance
                for j in classes[classI]:
                    if totalDistances[i, j] < distance:
                        distance = totalDistances[i, j]
                if distance < classDistance:
                    classes[classI].append(i)
                    newClass = False
                    break

            if newClass:
                classNum += 1
                classes.append([i])

        newClasses = [classes[0]]

        for subClass in classes[1:]:
            newClass = True
            for newClassI in range(len(newClasses)):
                distance = classDistance
                for i in newClasses[newClassI]:
                    for j in subClass:
                        if totalDistances[i, j] < distance:
                            distance = totalDistances[i, j]
                if distance < classDistance:
                    newClasses[newClassI] += subClass
                    newClass = False
                    break

            if newClass:
                newClasses.append(subClass)
    
        return newClasses

    def get_classes_centers(self):
        centers = self.centers
        classes = self._calc_classes(
            centers, self.classDistance, self.adj_distance(centers, centers[:, np.newaxis])
        )
        return {i + 1: classes[i] for i in range(len(classes))}, centers

    def plot_spatial(self, ax: plt.Axes = None, oscis: np.ndarray = None, index: int = -1, **kwargs):
        positionX, phaseTheta, _ = self.get_state(index)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        if oscis is None:
            oscis = np.arange(self.model.agentsNum)
        ax.scatter(
            positionX[oscis, 0], positionX[oscis, 1],
            s=100 * self.model.diameter / 0.1 * (2.5 / self.model.boundaryLength), 
            color=["#F8B08E"] * self.model.halfAgentsNum + ["#9BD5D5"] * self.model.halfAgentsNum,
            **kwargs
        )
        ax.quiver(
            positionX[oscis, 0], positionX[oscis, 1],
            np.cos(phaseTheta[oscis]), np.sin(phaseTheta[oscis]), 
            color=["#F16623"] * self.model.halfAgentsNum + ["#49B2B2"] * self.model.halfAgentsNum, 
            width=0.006, scale=18,
            **kwargs
        )
        ax.set_xlim(0, self.model.boundaryLength)
        ax.set_ylim(0, self.model.boundaryLength)    

    def plot_centers(self, ax: plt.Axes = None, index: int = -1):
        positionX, phaseTheta, _ = self.get_state(index)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        quiverColors = ["#FF4B4E"] * 500 + ["#414CC7"] * 500
        ax.quiver(
            positionX[:, 0], positionX[:, 1],
            np.cos(phaseTheta[:]), np.sin(phaseTheta[:]), color=quiverColors, alpha=0.8
        )
        centerColors = ["#FBDD85"] * 500 + ["#80A6E2"] * 500
        centers = self.centers
        ax.scatter(centers[:, 0], centers[:, 1], color=centerColors, s=5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)    
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
        ax.grid(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.set_xlabel(r"$x$", fontsize=16)
        ax.set_ylabel(r"$y$", fontsize=16, rotation=0)
    
    def calc_order_parameter_R(self, state: Tuple[np.ndarray, np.ndarray, np.ndarray] = None):
        if state is None:
            _, phaseTheta, _ = self.get_state(self.lookIndex)
        else:
            _, phaseTheta, _ = state

        return np.abs(np.sum(np.exp(1j * phaseTheta))) / phaseTheta.size