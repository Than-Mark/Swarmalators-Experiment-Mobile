import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from itertools import product
from typing import List, Tuple, Dict, Any
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

colors = ["#403990", "#3A76D6", "#FFC001", "#F46F43", "#FF0000"]
cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors)

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
)

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
        self.randomSeed = randomSeed
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
            posOmega = np.abs(np.random.standard_cauchy(agentsNum // 2))
        self.freqOmega = np.concatenate([
            posOmega, -posOmega
        ])
        self.freqOmega = np.sort(self.freqOmega)
        self.halfBoundaryLength = boundaryLength / 2
        self.counts = 0
        self.dotThetaParams = (
            self.boundaryLength,
            self.halfBoundaryLength,
            self.distanceD0,
            self.strengthK,
            self.phaseLagA0,
        )
    
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
        # return self._calc_dot_phase(self.deltaTheta, self.A, self.freqOmega, 
        #                             self.strengthK, self.phaseLagA0)
        return self._calc_dot_phase2(
                positionX=self.positionX, 
                phaseTheta=self.phaseTheta, 
                freqOmega=self.freqOmega, 
                params=self.dotThetaParams
            )
    
    @staticmethod
    @nb.njit
    def _calc_dot_phase2(positionX: np.ndarray, phaseTheta: np.ndarray, 
                         freqOmega: np.ndarray, params: Tuple[float]) -> np.ndarray:
        agentsNum = positionX.shape[0]
        boundaryLength, halfBoundaryLength, distanceD0, strengthK, phaseLagA0 = params

        coupling = np.zeros(agentsNum)
        for i in range(agentsNum):
            xDiff = np.abs(positionX[:, 0] - positionX[i, 0])
            yDiff = np.abs(positionX[:, 1] - positionX[i, 1])
            neighborIdxs = np.where(
                (xDiff < distanceD0) | (boundaryLength - xDiff < distanceD0) & 
                (yDiff < distanceD0) | (boundaryLength - yDiff < distanceD0)
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

            deltaTheta = phaseTheta[neighborIdxs][A] - phaseTheta[i]
            coupling[i] = np.mean(
                np.sin(deltaTheta + phaseLagA0)
            ) - np.sin(phaseLagA0)
        return strengthK * coupling + freqOmega

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
    def A(self) -> np.ndarray:
        """Adjacency matrix: 1 if |x_i - x_j| <= d0 else 0"""
        return np.where(self.distance_x(self.deltaX) <= self.distanceD0, 1, 0)

    @staticmethod
    @nb.njit
    def _calc_dot_phase(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, 
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
    
    def plot(self, ax: plt.Axes = None, colorsBy: str = "freq"):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        
        if colorsBy == "freq":
            colors = (
                ["red"] * (self.freqOmega >= 0).sum() + 
                ["#414CC7"] * (self.freqOmega < 0).sum()
            )
        elif colorsBy == "phase":
            colors = [cmap(i) for i in
                np.floor(256 - self.phaseTheta / (2 * np.pi) * 256).astype(np.int32)
            ]

        plt.quiver(
            self.positionX[:, 0], self.positionX[:, 1],
            np.cos(self.phaseTheta), np.sin(self.phaseTheta), 
            color=colors
        )
        plt.xlim(0, self.boundaryLength)
        plt.ylim(0, self.boundaryLength)

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"strengthK={self.strengthK:.3f},distanceD0={self.distanceD0:.3f},"
            f"phaseLagA0={self.phaseLagA0:.3f},boundaryLength={self.boundaryLength:.1f},"
            f"speedV={self.speedV:.1f},freqDist='{self.freqDist}',"
            f"{'initPhaseTheta,' if self.initPhaseTheta is not None else ''}"
            f"omegaMin={self.omegaMin:.3f},deltaOmega={self.deltaOmega:.3f},"
            f"agentsNum={self.agentsNum},dt={self.dt:.2f},"
            f"randomSeed={self.randomSeed}"
            ")"
        )
    

class PhaseLagPatternFormationNoPeriodic(PhaseLagPatternFormation):
    def update(self):
        dotPos = self.dotPosition
        dotPhase = self.dotPhase
        
        self.positionX = np.clip(
            self.positionX + dotPos * self.dt, 
            0, self.boundaryLength
        )
        self.phaseTheta = np.mod(self.phaseTheta + dotPhase * self.dt, 2 * np.pi)
    
    @property
    def deltaX(self) -> np.ndarray:
        return self.positionX - self.positionX[:, np.newaxis]


class PhaseLagPatternFormationNoCounter(PhaseLagPatternFormation):
    @staticmethod
    @nb.njit
    def _calc_dot_phase(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, 
                        K: float, phaseLagA0: float) -> np.ndarray:
        coupling = np.zeros(deltaTheta.shape[0])
        for idx in range(deltaTheta.shape[0]):
            coupling[idx] = np.mean(
                np.sin(deltaTheta[idx][A[idx] == 1] + phaseLagA0)
            )
        return K * coupling + omega


class AdditivePhaseLagPatternFormation(PhaseLagPatternFormation):
    @staticmethod
    @nb.njit
    def _calc_dot_phase2(positionX: np.ndarray, phaseTheta: np.ndarray, 
                         freqOmega: np.ndarray, params: Tuple[float]) -> np.ndarray:
        agentsNum = positionX.shape[0]
        boundaryLength, halfBoundaryLength, distanceD0, strengthK, phaseLagA0 = params

        coupling = np.zeros(agentsNum)
        for i in range(agentsNum):
            xDiff = np.abs(positionX[:, 0] - positionX[i, 0])
            yDiff = np.abs(positionX[:, 1] - positionX[i, 1])
            neighborIdxs = np.where(
                (xDiff < distanceD0) | (boundaryLength - xDiff < distanceD0) & 
                (yDiff < distanceD0) | (boundaryLength - yDiff < distanceD0)
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

            deltaTheta = phaseTheta[neighborIdxs][A] - phaseTheta[i]
            coupling[i] = np.sum(
                np.sin(deltaTheta + phaseLagA0) - np.sin(phaseLagA0)
            )
        return strengthK * coupling + freqOmega


class OnlyCounterPhaseLagPatternFormation(PhaseLagPatternFormation):
    @staticmethod
    @nb.njit
    def _calc_dot_phase(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, 
                        K: float, phaseLagA0: float) -> np.ndarray:
        coupling = np.zeros(deltaTheta.shape[0])
        for idx in range(deltaTheta.shape[0]):
            coupling[idx] = np.mean(
                np.sin(deltaTheta[idx][A[idx] == 1]) - np.sin(phaseLagA0)
            )
        return K * coupling + omega


class PurePhaseFrustration(PhaseLagPatternFormation):
    def __init__(self, strengthK: float, phaseLagA0: float, 
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None, 
                 omegaMin: float = 0.1, deltaOmega: float = 1, 
                 agentsNum: int = 1000, dt: float = 0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10, 
                 randomSeed: int = 10, overWrite: bool = False):
        super().__init__(strengthK, 0, phaseLagA0, 0, 0, 
                         freqDist, initPhaseTheta, 
                         omegaMin, deltaOmega, 
                         agentsNum, dt, 
                         tqdm, savePath, shotsnaps, 
                         randomSeed, overWrite)
        
    @property
    def dotPhase(self) -> np.ndarray:
        return self._calc_dot_phase(self.deltaTheta, None, self.freqOmega, 
                                    self.strengthK, self.phaseLagA0)

    @staticmethod
    @nb.njit
    def _calc_dot_phase(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, 
                        K: float, phaseLagA0: float) -> np.ndarray:
        coupling = np.zeros(deltaTheta.shape[0])
        for idx in range(deltaTheta.shape[0]):
            coupling[idx] = np.mean(
                np.sin(deltaTheta[idx] + phaseLagA0) - np.sin(phaseLagA0)
            )
        return K * coupling + omega

    def update(self):
        self.phaseTheta = np.mod(self.phaseTheta + self.dotPhase * self.dt, 2 * np.pi)

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))


class PhaseLagPatternFormation1D(PhaseLagPatternFormation):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0.1, deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        super().__init__(strengthK, distanceD0, phaseLagA0,
                         boundaryLength, speedV, freqDist, initPhaseTheta,
                         omegaMin, deltaOmega, agentsNum, dt,
                         tqdm, savePath, shotsnaps, randomSeed, overWrite)

        self.positionX = np.random.random(agentsNum) * boundaryLength

    @staticmethod
    @nb.njit
    def _direction(phaseTheta: np.ndarray) -> np.ndarray:
        return np.cos(phaseTheta)
    
    @property
    def A(self) -> np.ndarray:
        """Adjacency matrix: 1 if |x_i - x_j| <= d0 else 0"""
        return np.where(np.abs(self.deltaX) <= self.distanceD0, 1, 0)
    
    def plot(self, ax: plt.Axes = None) -> None:
        colors = [new_cmap(i) for i in
            np.floor(256 - self.phaseTheta / (2 * np.pi) * 256).astype(np.int32)
        ]

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        plt.quiver(
            self.positionX, np.zeros(self.agentsNum),
            np.cos(self.phaseTheta), np.sin(self.phaseTheta), 
            color=colors, scale=25, width=0.005,
        )

        plt.plot([0, self.boundaryLength], [0, 0], color="black", lw=1.5)
        plt.xlim(-0.1, 0.1 + self.boundaryLength)
        plt.ylim(-0.9, 0.9)
        plt.yticks([])

        ax.set_aspect('equal', adjustable='box')
        for line in ["top", "right"]:
            ax.spines[line].set_visible(False)

        plt.grid()
        plt.tick_params(direction='in')
        plt.scatter(np.full(self.agentsNum, -2), np.full(self.agentsNum, -2),
                    c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
        plt.colorbar(ticks=[0, np.pi, 2*np.pi], ax=ax).ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])


class StateAnalysis:
    def __init__(self, model: PhaseLagPatternFormation = None):
        if model is None:
            return
        self.model = model
        
        targetPath = f"{self.model.savePath}/{self.model}.h5"
        
        totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
        TNum = totalPhaseTheta.shape[0] // self.model.agentsNum
        self.TNum = TNum
        self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)

        if isinstance(model, PurePhaseFrustration):
            return

        totalPositionX = pd.read_hdf(targetPath, key="positionX")
        if isinstance(model, PhaseLagPatternFormation1D):
            self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum)
        else:
            self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
        

    def get_state(self, index: int = -1):
        if isinstance(self.model, PurePhaseFrustration):
            positionX = None
        else:
            positionX = self.totalPositionX[index]
        phaseTheta = self.totalPhaseTheta[index]

        return positionX, phaseTheta
    
    def plot_spatial(self, ax: plt.Axes = None, 
                     colorsBy: str = "freq", index: int = -1, 
                     shift: np.ndarray = np.array([0, 0])):
        assert colorsBy in ["freq", "phase"], "colorsBy must be 'freq' or 'phase'"

        if isinstance(self.model, PhaseLagPatternFormation1D):
            self.plot_spatial_1D(ax, colorsBy, index)
        else:
            self.plot_spatial_2D(ax, colorsBy, index, shift)

    def plot_spatial_2D(self, ax: plt.Axes = None, 
                     colorsBy: str = "freq", index: int = -1, 
                     shift: np.ndarray = np.array([0, 0])):

        positionX, phaseTheta = self.get_state(index)
        positionX = np.mod(positionX + shift, self.model.boundaryLength)

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        if colorsBy == "freq":
            colors = (
                ["red"] * (self.model.freqOmega >= 0).sum() + 
                ["#414CC7"] * (self.model.freqOmega < 0).sum()
            )
        elif colorsBy == "phase":
            colors = [cmap(i) for i in
                np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)
            ]

        ax.quiver(
            positionX[:, 0], positionX[:, 1],
            np.cos(phaseTheta), np.sin(phaseTheta), 
            scale_units='inches', scale=15.0, width=0.002,
            color=colors
        )
        ax.set_xlim(0, self.model.boundaryLength)
        ax.set_ylim(0, self.model.boundaryLength)

    def plot_spatial_1D(self, ax: plt.Axes = None, 
                        colorsBy: str = "freq", index: int = -1):
        positionX, phaseTheta = self.get_state(index)

        colors = [new_cmap(i) for i in
            np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)
        ]

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        plt.quiver(
            positionX, np.zeros(self.model.agentsNum),
            np.cos(phaseTheta), np.sin(phaseTheta), 
            color=colors, scale=25, width=0.005,
        )

        plt.plot([0, self.model.boundaryLength], [0, 0], color="black", lw=1.5)
        plt.xlim(-0.1, 0.1 + self.model.boundaryLength)
        plt.ylim(-0.9, 0.9)
        plt.yticks([])

        ax.set_aspect('equal', adjustable='box')
        for line in ["top", "right"]:
            ax.spines[line].set_visible(False)

        plt.grid()
        plt.tick_params(direction='in')
        plt.scatter(np.full(self.model.agentsNum, -2), np.full(self.model.agentsNum, -2),
                    c=phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
        plt.colorbar(ticks=[0, np.pi, 2*np.pi], ax=ax).ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])

    def check_state_input(self, positionX: np.ndarray = None, phaseTheta: np.ndarray = None,
                          lookIdx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        if ((positionX is None and phaseTheta is not None) or 
            (positionX is not None and phaseTheta is None)):
            raise ValueError("Both positionX and phaseTheta must be provided or both must be None.")
        if positionX is None:
            positionX, phaseTheta = self.get_state(lookIdx)
        return positionX, phaseTheta

    def calc_dot_theta(self, positionX: np.ndarray = None, phaseTheta: np.ndarray = None,
                       lookIdx: int = -1) -> np.ndarray:
        positionX, phaseTheta = self.check_state_input(positionX, phaseTheta, lookIdx)
    
        deltaTheta = phaseTheta - phaseTheta[:, np.newaxis]
        deltaX = self.model._delta_x(positionX, positionX[:, np.newaxis], 
                                    self.model.boundaryLength, self.model.halfBoundaryLength)
        A = np.where(self.model.distance_x(deltaX) <= self.model.distanceD0, 1, 0)
        return self.model._calc_dot_phase(deltaTheta, A, self.model.freqOmega, 
                                        self.model.strengthK, self.model.phaseLagA0)
    
    def calc_rotation_center(self, positionX: np.ndarray = None, phaseTheta: np.ndarray = None,
                       lookIdx: int = -1) -> np.ndarray:
        positionX, phaseTheta = self.check_state_input(positionX, phaseTheta, lookIdx)

        positionx, positiony = positionX[:, 0], positionX[:, 1]
        dotPhase = self.calc_dot_theta(positionX, phaseTheta)

        return np.array([
            positionx - self.model.speedV / dotPhase * np.sin(phaseTheta),
            positiony + self.model.speedV / dotPhase * np.cos(phaseTheta)
        ]).T
    
    def calc_classes_and_centers(self, classDistance: float = 0.1,
                                 positionX: np.ndarray = None,
                                 phaseTheta: np.ndarray = None,
                                 lookIdx: int = -1) -> Tuple[List[List[int]], np.ndarray]:
        positionX, phaseTheta = self.check_state_input(positionX, phaseTheta, lookIdx)
        
        centers = self.calc_rotation_center(positionX, phaseTheta, lookIdx)
        centers = np.mod(centers, self.model.boundaryLength)
        totalDistances = self.model.distance_x(self.model._delta_x(
            centers, centers[:, np.newaxis], 
            self.model.boundaryLength, self.model.halfBoundaryLength
        ))

        classes = self._calc_classes(centers, classDistance, totalDistances)
        return classes, centers
    
    def calc_classes(self, classDistance: float = 0.1,
                     positionX: np.ndarray = None,
                     phaseTheta: np.ndarray = None,
                     lookIdx: int = -1) -> List[List[int]]:
        classes, _ = self.calc_classes_and_centers(
            classDistance, positionX, phaseTheta, lookIdx
        )
        return classes

    @staticmethod
    @nb.njit
    def _calc_classes(centers: np.ndarray, classDistance: float, totalDistances: np.ndarray) -> List[List[int]]:
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
    
    def calc_replative_distance(self, position1: np.ndarray, position2: np.ndarray) -> float:
        deltaX = self.model._delta_x(position1, position2, 
                                     self.model.boundaryLength, 
                                     self.model.halfBoundaryLength)
        return np.linalg.norm(deltaX, axis=-1)

    def calc_nearby_edges(self, edgeLenThres: float, classCenters: np.ndarray,
                          relativeDistance: bool = False) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        if relativeDistance:
            classIdxs = np.arange(len(classCenters))
            edges = np.unique([
                np.sort(adj) for adj in product(classIdxs, classIdxs) if adj[0] != adj[1]
            ], axis=0)
            edges = [
                edge for edge in edges
                if self.calc_replative_distance(
                    classCenters[edge[0]], classCenters[edge[1]]
                ) < edgeLenThres
            ]
            return [tuple(edge) for edge in edges]
        else:
            # For plot in periodic boundary conditions
            # classCenters be adjusted to include periodic images
            positionShifts = product(
                [-self.model.boundaryLength, 0, self.model.boundaryLength],
                [-self.model.boundaryLength, 0, self.model.boundaryLength]
            )
            periodicCenters = []
            for xShift, yShift in positionShifts:
                periodicCenters.append(
                    np.array([classCenters[:, 0] + xShift, classCenters[:, 1] + yShift]).T
                )
            classCenters = np.concatenate(periodicCenters, axis=0)

            classIdxs = np.arange(len(classCenters))
            edges = np.unique([
                np.sort(adj) for adj in product(classIdxs, classIdxs) if adj[0] != adj[1]
            ], axis=0)
            edges = [
                edge for edge in edges
                if np.linalg.norm(
                    classCenters[edge[0]] - classCenters[edge[1]]
                ) < edgeLenThres
            ]
            return [tuple(edge) for edge in edges], classCenters
        
    def select_classIdx_of_line(self, selectClassIdx: int, classCenters: np.ndarray,
                                visualAngle: float, span: float) -> List[int]:
        selectClassPos = classCenters[selectClassIdx]
        deltaX = self.model._delta_x(selectClassPos, classCenters, 
                                    self.model.boundaryLength, 
                                    self.model.halfBoundaryLength)
        spaceAngle = np.arctan2(deltaX[:, 1], deltaX[:, 0])
        filterClassIdx = np.where(
            (np.abs(spaceAngle - visualAngle) < span) |
            (np.abs(spaceAngle + np.pi  - visualAngle) < span)
        )[0]
        return filterClassIdx.tolist() + [selectClassIdx]
    
    def calc_order_parameter_R(self, phaseTheta: np.ndarray = None,
                               lookIdx: int = -1) -> float:
        if phaseTheta is None:
            _, phaseTheta = self.get_state(lookIdx)
        
        return np.abs(np.mean(np.exp(1j * phaseTheta)))