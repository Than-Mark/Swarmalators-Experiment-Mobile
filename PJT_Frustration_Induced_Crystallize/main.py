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

colors = ["#403990", "#80A6E2", "#FBDD85", "#F46F43", "#CF3D3E"]
cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors)


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
        return self._calc_dot_phase(self.deltaTheta, self.A, self.freqOmega, 
                                    self.strengthK, self.phaseLagA0)

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
    
    def plot(self, ax: plt.Axes = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        
        colors = ["red"] * (self.freqOmega >= 0).sum() + ["#414CC7"] * (self.freqOmega < 0).sum()

        plt.quiver(
            self.positionX[:, 0], self.positionX[:, 1],
            np.cos(self.phaseTheta), np.sin(self.phaseTheta), 
            color=colors
        )
        plt.xlim(0, self.boundaryLength)
        plt.ylim(0, self.boundaryLength)

    def __str__(self):
        return (
            f"{self.__class__.__name__}(strengthK={self.strengthK:.3f},distanceD0={self.distanceD0:.3f},"
            f"phaseLagA0={self.phaseLagA0:.3f},boundaryLength={self.boundaryLength:.1f},"
            f"speedV={self.speedV:.1f},freqDist='{self.freqDist}',omegaMin={self.omegaMin:.3f},"
            f"deltaOmega={self.deltaOmega:.3f},agentsNum={self.agentsNum},dt={self.dt:.2f}),"
            f"randomSeed={self.randomSeed}"
        )
    

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
    
    def plot_spatial(self, ax: plt.Axes = None, 
                     colorsBy: str = "freq", index: int = -1, 
                     shift: np.ndarray = np.array([0, 0])):
        assert colorsBy in ["freq", "phase"], "colorsBy must be 'freq' or 'phase'"

        positionX, phaseTheta = self.get_state(index)
        positionX = np.mod(positionX + shift, self.model.boundaryLength)

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        if colorsBy == "freq":
            colors = ["red"] * (self.model.freqOmega >= 0).sum() + ["#414CC7"] * (self.model.freqOmega < 0).sum()
        elif colorsBy == "phase":
            colors = [cmap(i) for i in
                np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)
            ]

        ax.quiver(
            positionX[:, 0], positionX[:, 1],
            np.cos(phaseTheta), np.sin(phaseTheta), 
            color=colors
        )
        ax.set_xlim(0, self.model.boundaryLength)
        ax.set_ylim(0, self.model.boundaryLength)

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