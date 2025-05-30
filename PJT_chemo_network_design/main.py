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
        self.u = np.random.rand(cellNumInLine, cellNumInLine)
        self.v = np.random.rand(cellNumInLine, cellNumInLine)
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

    @staticmethod
    @nb.njit
    def _short_rep(positionX: np.ndarray, diameter: float,
                   halfBoundaryLength: float, boundaryLength: float, 
                   power: float, cutOff: bool):
        if not cutOff:
            diameter = np.inf
        rep = np.zeros(positionX.shape)
        for i in range(positionX.shape[0]):
            neighbor = positionX[
                (np.abs(positionX[:, 0] - positionX[i, 0]) % halfBoundaryLength < diameter)
                & (np.abs(positionX[:, 1] - positionX[i, 1]) % halfBoundaryLength < diameter)
                & ((positionX[:,0] != positionX[i,0]) | (positionX[:, 1] != positionX[i, 1]))
            ]
            if neighbor.shape[0]== 0:
                continue
            subX = positionX[i] - neighbor
            deltaX =(
                subX * (-halfBoundaryLength<= subX) * (subX<= halfBoundaryLength) + 
                (subX + boundaryLength) * (subX < -halfBoundaryLength)+
                (subX - boundaryLength) * (subX > halfBoundaryLength)
            )
            distance = np.sqrt(deltaX[:, 0] ** 2 + deltaX[:,1] ** 2).reshape(-1, 1)
            if cutOff:            
                rep[i] = np.sum(deltaX / distance ** power * (distance < diameter), axis=0)
            else:
                rep[i] = np.sum(deltaX / distance ** power, axis=0)
        
        return rep
    
    @staticmethod
    @nb.njit
    def _product_c(cellNumInLine: int, ocsiIdx: np.ndarray, productRateK0: float):
        productC = np.zeros((cellNumInLine, cellNumInLine), dtype=np.float64)
        for idx in ocsiIdx:
            productC[idx[0], idx[1]] = productC[idx[0], idx[1]] + 1
        return productC * productRateK0


class PathPlanningGSC(PatternFormation):
    def __init__(self, 
                 nodePosition: np.ndarray, 
                 productRateBetac: float = 1, decayRateKc: float = 0.001, 
                 diffusionRateDc: float = 1, convectionVc: float = 1, 
                 cDecayBase: float = 0.8, cControlThres: float = 0.5,
                 noiseRateBetaDp: float = 0.1, initialState: float = 0., chemoAlphaC: float = 1,
                 diameter: float = 3, repelPower: float = 2, repCutOff: bool = True, deltaSpread: bool = False,
                 cellNumInLine: int = 200, boundaryLength: float = 200, 
                 agentsNum: int=1000, dt: float=0.1, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 100, 
                 randomSeed: int = 10, overWrite: bool = False) -> None:

        self.halfAgentsNum = agentsNum // 2

        np.random.seed(randomSeed)
        self.nodePosition = nodePosition
        # self.positionX = np.random.random((agentsNum, 2)) * boundaryLength / 4 * 3 + boundaryLength / 8
        # self.positionX = np.random.random((agentsNum, 2)) * boundaryLength / 2 + boundaryLength / 4
        # self.positionX = np.random.random((agentsNum, 2)) * boundaryLength / 4 + boundaryLength / 8 * 3
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.initialState = initialState
        if initialState is None:
            self.internalState = np.random.rand(agentsNum) * 2 - 1
        else:
            self.internalState = np.ones(agentsNum) * initialState
        self.cellNumInLine = cellNumInLine
        self.cPosition = np.array(list(product(np.linspace(0, boundaryLength, cellNumInLine), repeat=2)))
        self.dx = boundaryLength / (cellNumInLine - 1)
        self.agentsNum = agentsNum
        self.decayRateKc = decayRateKc
        self.diffusionRateDc = diffusionRateDc
        self.dt = dt
        self.diameter = diameter
        self.repelPower = repelPower
        self.repCutOff = repCutOff
        self.convectionVc = convectionVc
        self.cDecayBase = cDecayBase
        self.cControlThres = cControlThres
        self.deltaSpread = deltaSpread
        self.spreadNum = int(np.round((self.diameter / self.dx - 1) / 2))

        self.noiseRateBetaDp = noiseRateBetaDp
        self.noiseMultiAdj = np.sqrt(2 * self.noiseRateBetaDp)

        if deltaSpread:
            productAdjMulti = boundaryLength**2 / (cellNumInLine * (1 + 2 * self.spreadNum))**2
        else:
            productAdjMulti = boundaryLength**2 / cellNumInLine**2
        self.productRateBetac = productRateBetac
        self._productRateBetac = productRateBetac * productAdjMulti
        
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.boundaryLength = boundaryLength
        self.halfBoundaryLength = boundaryLength / 2
        self.randomSeed = randomSeed
        self.overWrite = overWrite

        self.c = np.zeros((cellNumInLine, cellNumInLine))
        self.chemoAlphaC = chemoAlphaC

        self.temp = dict()
        # The order of variable definitions has a dependency relationship
        self.temp["ocsiIdx"] = (self.positionX / self.dx).round().astype(int)
        self.temp["nodeIdx"] = (self.nodePosition / self.dx).round().astype(int)
        if len(nodePosition) > 0:
            self.temp["productDeltaNode"] = self.calc_product_delta(idxKey="nodeIdx") 
        else:
            self.temp["productDeltaNode"] = np.zeros((cellNumInLine, cellNumInLine))

    @property
    def nablaC(self):
        return - np.array([ 
            (np.roll(self.c, -1, axis=0) - np.roll(self.c, 1, axis=0)),
            (np.roll(self.c, -1, axis=1) - np.roll(self.c, 1, axis=1))
        ]).transpose(1, 2, 0) / (2 * self.dx)
    
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
    def productC(self):
        return self.temp["productDeltaNode"] * self._productRateBetac
    
    @property
    def dotC(self):
        localState = self._product_c(self.cellNumInLine, self.temp["ocsiIdx"], self.internalState, 
                                     meaning=False, spreadNum=self.spreadNum)
        localState = np.clip(localState, 0, 1)

        diffX = self.positionX.mean(axis=0) - self.positionX
        phi = np.arctan2(diffX[:, 1], diffX[:, 0])
        localPhi = self._product_c(self.cellNumInLine, self.temp["ocsiIdx"], phi, 
                                     meaning=False, spreadNum=self.spreadNum)

        nablaC = self.nablaC

        convection = self.convectionVc * localState * (
            np.cos(localPhi) * nablaC[:, :, 0] + np.sin(localPhi) * nablaC[:, :, 1]
        )

        return (
            self.diffusionRateDc * self.nabla2C
            - convection
            - self.decayRateKc * self.c
            + self.productC
        )
    
    @property
    def dotInternalState(self):
        localC = self.c[self.temp["ocsiIdx"][:, 0], self.temp["ocsiIdx"][:, 1]]
        return self.internalState * (1 - self.internalState) * (localC - self.cControlThres)

    @property
    def localGradient(self):
        localGradC = self.nablaC[self.temp["ocsiIdx"][:, 0], self.temp["ocsiIdx"][:, 1]]
        return self.chemoAlphaC * localGradC

    @property
    def spatialNoise(self):
        noise = self.noiseMultiAdj * np.random.normal(
            loc=0, scale=1, size=(self.agentsNum, 2)
        )
        return noise

    @staticmethod
    @nb.njit
    def _short_rep(positionX: np.ndarray, diameter: float,
                   halfBoundaryLength: float, boundaryLength: float, 
                   power: float, cutOff: bool):
        if not cutOff:
            diameter = np.inf
        rep = np.zeros(positionX.shape)
        for i in range(positionX.shape[0]):
            neighborPos: np.ndarray = positionX[
                (np.abs(positionX[:, 0] - positionX[i, 0]) % halfBoundaryLength < diameter)
                & (np.abs(positionX[:, 1] - positionX[i, 1]) % halfBoundaryLength < diameter)
                & ((positionX[:,0] != positionX[i,0]) | (positionX[:, 1] != positionX[i, 1]))
            ]
            if neighborPos.shape[0]== 0:
                continue
            subX = positionX[i] - neighborPos
            deltaX =(
                subX * (-halfBoundaryLength<= subX) * (subX<= halfBoundaryLength) + 
                (subX + boundaryLength) * (subX < -halfBoundaryLength)+
                (subX - boundaryLength) * (subX > halfBoundaryLength)
            )
            distance = np.sqrt(deltaX[:, 0] ** 2 + deltaX[:,1] ** 2).reshape(-1, 1)
            if cutOff:            
                rep[i] = np.sum(deltaX / distance ** power * (distance < diameter), axis=0)
            else:
                rep[i] = np.sum(deltaX / distance ** power, axis=0)

        return rep

    @property
    def shortRepulsion(self):
        return self._short_rep(
            self.positionX, self.diameter,
            self.halfBoundaryLength, self.boundaryLength, 
            self.repelPower, self.repCutOff
        )

    def calc_product_delta(self, idxKey: str):
        if idxKey == "ocsiIdx":
            return self._product_c(self.cellNumInLine, self.temp[idxKey], self.temp["neighborsNum"], 
                                   meaning=False, spreadNum=self.spreadNum)
        else:  # nodeIdx
            return self._product_c(self.cellNumInLine, self.temp[idxKey], np.ones(self.agentsNum))

    def update(self):
        self.temp["ocsiIdx"] = (self.positionX / self.dx).round().astype(int)
        shortRepulsion = self.shortRepulsion
        if self.diameter == 0:
            dotPosition = self.localGradient + self.spatialNoise
        else:
            dotPosition = self.localGradient + self.spatialNoise + shortRepulsion
        dotInternalState = self.dotInternalState
        dotC = self.dotC
        
        self.positionX = np.mod(
            self.positionX + dotPosition * self.dt, 
            self.boundaryLength
        )
        self.internalState += dotInternalState * self.dt
        self.c += dotC * self.dt
        self.c[self.c < 0] = 0

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="internalState", value=pd.DataFrame(self.internalState))
            self.store.append(key="c", value=pd.DataFrame(self.c))

    def __str__(self) -> str:
        name =  (
            f"PPGS"
            f"_bC{self.productRateBetac:.3f}_kC{self.decayRateKc:.3f}_Dc{self.diffusionRateDc:.3f}"
            f"_cVc{self.convectionVc:.3f}"
            f"_cDB{self.cDecayBase:.3f}"
            f"_cCT{self.cControlThres:.3f}_aC{self.chemoAlphaC:.3f}_initS{self.initialState:.2f}"
            f"_d{self.diameter:.1f}_rep{self.repelPower:.1f}_delSp{self.deltaSpread}"
            f"_cutoff{self.repCutOff}_noise{self.noiseRateBetaDp:.3f}"
            f"_cell{self.cellNumInLine}_L{self.boundaryLength:.1f}"
            f"_agents{self.agentsNum}_nodes{self.nodePosition.shape[0]}"
            f"_r{self.randomSeed}_dt{self.dt:.3f}"
        )
        
        return name
    

class PathPlanningGSCA(PathPlanningGSC):
    def __init__(self, nodePosition, productRateBetac = 1, decayRateKc = 0.001, 
                 diffusionRateDc = 1, convectionVc = 1, 
                 cDecayBase = 0.8, cControlThres = 0.5, 
                 initialState = None, 
                 stateSenseSpeed: float = 0.1, senseDistence: float = 20,
                 noiseRateBetaDp = 0.1, chemoAlphaC = 1, 
                 diameter = 3, repelPower = 2, repCutOff = True, deltaSpread = False, 
                 cellNumInLine = 200, boundaryLength = 200, 
                 agentsNum = 1000, dt = 0.1, 
                 tqdm = False, savePath = None, shotsnaps = 100, 
                 randomSeed = 10, overWrite = False):
        super().__init__(nodePosition, productRateBetac, decayRateKc, diffusionRateDc, convectionVc, cDecayBase, cControlThres, noiseRateBetaDp, initialState, chemoAlphaC, diameter, repelPower, repCutOff, deltaSpread, cellNumInLine, boundaryLength, agentsNum, dt, tqdm, savePath, shotsnaps, randomSeed, overWrite)
        
        if initialState is None:
            self.internalState = np.random.choice([-0.99, 0.99], size=nodePosition.shape[0], p=[0.5, 0.5])
        else:
            self.internalState = np.ones(nodePosition.shape[0]) * initialState

        self.stateSenseSpeed = stateSenseSpeed
        self.senseDistence = senseDistence
        self.senseSpreadNum = int(np.round((self.senseDistence / self.dx - 1) / 2))

    def __str__(self) -> str:
        name =  (
            f"PPGA"
            f"_bC{self.productRateBetac:.3f}_kC{self.decayRateKc:.3f}_Dc{self.diffusionRateDc:.3f}"
            f"_sD{self.senseDistence:.1f}_sSS{self.stateSenseSpeed:.3f}"
            f"_cDB{self.cDecayBase:.3f}"
            f"_cCT{self.cControlThres:.3f}_aC{self.chemoAlphaC:.3f}"
            f"_initS{self.initialState:.2f}" if self.initialState is not None else ""
            f"_d{self.diameter:.1f}_rep{self.repelPower:.1f}_delSp{self.deltaSpread}"
            f"_cutoff{self.repCutOff}_noise{self.noiseRateBetaDp:.3f}"
            f"_cell{self.cellNumInLine}_L{self.boundaryLength:.1f}"
            f"_agents{self.agentsNum}_nodes{self.nodePosition.shape[0]}"
            f"_r{self.randomSeed}_dt{self.dt:.3f}"
        )
        
        return name

    @property
    def dotC(self):
        productC = self._product_c(self.cellNumInLine, self.temp["nodeIdx"], self.internalState,
                                   meaning=False, spreadNum=self.spreadNum)
        return (
            self.diffusionRateDc * self.nabla2C
            - self.decayRateKc * self.c
            + productC
        )
    
    @property
    def dotInternalState(self):
        globalSenseCounts = self._product_c(
            self.cellNumInLine, self.temp["ocsiIdx"], np.ones(self.agentsNum),
            meaning=False, spreadNum=self.senseSpreadNum
        )
        localCounts = globalSenseCounts[self.temp["nodeIdx"][:, 0], self.temp["nodeIdx"][:, 1]]
        return self.stateSenseSpeed * (1 + self.internalState) * (1 - self.internalState) * (3 - localCounts)


class StateAnalysis:
    def __init__(self, model: PathPlanningGSC = None, classDistance: float = 2, 
                 lookIndex: int = -1, showTqdm: bool = False):
        
        self.classDistance = classDistance
        self.lookIndex = lookIndex
        self.showTqdm = showTqdm
        
        if model is not None:
            self.model = model
            targetPath = f"{self.model.savePath}/{self.model}.h5"
            totalPositionX = pd.read_hdf(targetPath, key="positionX")
            totalInternalState = pd.read_hdf(targetPath, key="internalState")
            totalC = pd.read_hdf(targetPath, key="c")
            
            TNum = totalPositionX.shape[0] // self.model.agentsNum
            self.TNum = TNum
            self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
            self.totalInternalState = totalInternalState.values.reshape(TNum, self.model.nodePosition.shape[0])
            self.totalC = totalC.values.reshape(TNum, model.cellNumInLine, model.cellNumInLine)
            
            self.maxC = self.totalC[-1].max()
            self.minC = self.totalC[-1].min()

    def get_state(self, index: int = -1):
        positionX = self.totalPositionX[index]
        internalState = self.totalInternalState[index]
        c = self.totalC[index]

        return positionX, internalState, c

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
    
    def plot_spatial(self, ax: plt.Axes = None, oscis: np.ndarray = None, index: int = -1, **kwargs):
        positionX = self.get_state(index)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        if oscis is None:
            oscis = np.arange(self.model.agentsNum)
        for i in oscis:
            ax.add_artist(plt.Circle(
                positionX[i], self.model.diameter / 2 * 0.95, zorder=1, 
                facecolor="#9BD5D5", edgecolor="black"
            ))

        ax.set_xlim(0, self.model.boundaryLength)
        ax.set_ylim(0, self.model.boundaryLength)

    # def plot_single_field(self)

    def plot_fields(self, ax: plt.Axes = None, index: int = -1, 
                    fixExtremum: bool = False, withStream: bool = False):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 7))
        c1: np.ndarray = self.totalU[index]
        c2: np.ndarray = self.totalV[index]
        if fixExtremum:
            vmaxC1, vmaxC2, vminC1, vminC2 = self.maxC1, self.maxC2, self.minC1, self.minC2
        else:
            vmaxC1 = vmaxC2 = vminC1 = vminC2 = None

        if c1.mean() > c2.mean():
            pc1 = ax.imshow(self.totalU[index].T, cmap=self.c1Maps, vmax=vmaxC1, vmin=vminC1)
            pc2 = ax.imshow(self.totalV[index].T, cmap=self.c2Maps, vmax=vmaxC2, vmin=vminC2)
        else:
            pc2 = ax.imshow(self.totalV[index].T, cmap=self.c2Maps, vmax=vmaxC2, vmin=vminC2)
            pc1 = ax.imshow(self.totalU[index].T, cmap=self.c1Maps, vmax=vmaxC1, vmin=vminC1)

        plt.colorbar(pc1)
        plt.colorbar(pc2)

        if withStream:
            adjMulti = self.model.cellNumInLine / self.model.boundaryLength
            cPosition = self.model.cPosition.reshape(self.model.cellNumInLine, self.model.cellNumInLine, 2)
            X = cPosition[:, :, 0].T * adjMulti
            Y = cPosition[:, :, 1].T * adjMulti
            c = c1.T
            U = np.roll(c, -1, axis=1) - np.roll(c, 1, axis=1)
            V = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
            ax.streamplot(X, Y, U, V, color="white", linewidth=1, density=1.5)
            c = c2.T
            U = np.roll(c, -1, axis=1) - np.roll(c, 1, axis=1)
            V = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
            ax.streamplot(X, Y, U, V, color="black", linewidth=1, density=1.5)

        plt.xlim(0, self.model.cellNumInLine)
        plt.ylim(0, self.model.cellNumInLine)

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