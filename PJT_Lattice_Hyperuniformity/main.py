import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Iterable
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
    from tqdm.notebook import tqdm
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
from swarmalatorlib.template import Swarmalators2D


class PhaseLagPatternFormation(Swarmalators2D):
    def __init__(self, strengthK: float, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0., deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        
        assert freqDist in ["uniform", "cauchy"]
        
        if freqDist == "cauchy":
            omegaMin = 0.0
            deltaOmega = 0.0

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
    
    def plot(self, ax: plt.Axes = None, colorsBy: str = "phase"):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        
        if colorsBy == "freq":
            colors = (
                ["red"] * (self.freqOmega >= 0).sum() + 
                ["#414CC7"] * (self.freqOmega < 0).sum()
            )
        elif colorsBy == "phase":
            colors = [hexCmap(i) for i in
                np.floor(256 - self.phaseTheta / (2 * np.pi) * 256).astype(np.int32)
            ]

        ax.quiver(
            self.positionX[:, 0], self.positionX[:, 1],
            np.cos(self.phaseTheta), np.sin(self.phaseTheta), 
            scale_units='inches', scale=15.0, width=0.002,
            color=colors
        )
        ax.set_xlim(0, self.boundaryLength)
        ax.set_ylim(0, self.boundaryLength)

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"strengthK={self.strengthK:.3f},distanceD0={self.distanceD0:.3f},"
            f"phaseLagA0={self.phaseLagA0:.3f},boundaryLength={self.boundaryLength:.1f},"
            f"speedV={self.speedV:.1f},freqDist={self.freqDist},"
            f"{'initPhaseTheta,' if self.initPhaseTheta is not None else ''}"
            f"omegaMin={self.omegaMin:.3f},deltaOmega={self.deltaOmega:.3f},"
            f"agentsNum={self.agentsNum},dt={self.dt:.3f},"
            f"shotsnaps={self.shotsnaps},randomSeed={self.randomSeed}"
            ")"
        )
    

class VaryingStrengthK(PhaseLagPatternFormation):
    def __init__(self, strengthKRange: Iterable, distanceD0: float, phaseLagA0: float,
                 boundaryLength: float = 7, speedV: float = 3.0,
                 freqDist: str = "uniform", initPhaseTheta: np.ndarray = None,
                 omegaMin: float = 0., deltaOmega: float = 1.0,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 10,
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        super().__init__(strengthKRange[0], distanceD0, phaseLagA0, boundaryLength, speedV, 
                         freqDist, initPhaseTheta, omegaMin, deltaOmega, agentsNum, 
                         dt, tqdm, savePath, shotsnaps, randomSeed, overWrite)
        self.strengthKRange = strengthKRange
        
    def update(self):
        self.strengthK = self.strengthKRange[self.counts]

        dotPos = self.dotPosition
        dotPhase = self.dotPhase
        
        self.positionX = np.mod(self.positionX + dotPos * self.dt, self.boundaryLength)
        self.phaseTheta = np.mod(self.phaseTheta + dotPhase * self.dt, 2 * np.pi)

    def run(self):
        super().run(len(self.strengthKRange))

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"strengthK={self.strengthKRange[0]}to{self.strengthKRange[-1]}len{len(self.strengthKRange)},"
            f"distanceD0={self.distanceD0:.3f},"
            f"phaseLagA0={self.phaseLagA0:.3f},boundaryLength={self.boundaryLength:.1f},"
            f"speedV={self.speedV:.1f},freqDist={self.freqDist},"
            f"{'initPhaseTheta,' if self.initPhaseTheta is not None else ''}"
            f"omegaMin={self.omegaMin:.3f},deltaOmega={self.deltaOmega:.3f},"
            f"agentsNum={self.agentsNum},dt={self.dt:.3f},"
            f"shotsnaps={self.shotsnaps},randomSeed={self.randomSeed}"
            ")"
        )


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

        totalPositionX = pd.read_hdf(targetPath, key="positionX")
        self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
        

    def get_state(self, index: int = -1):

        positionX = self.totalPositionX[index]
        phaseTheta = self.totalPhaseTheta[index]

        return positionX, phaseTheta
    
    def plot_spatial(self, ax: plt.Axes = None, 
                     colorsBy: str = "phase", index: int = -1, 
                     shift: np.ndarray = np.array([0, 0])):
        assert colorsBy in ["freq", "phase"], "colorsBy must be 'freq' or 'phase'"

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
            colors = [hexCmap(i) for i in
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
        plt.colorbar(ticks=[0, np.pi, 2*np.pi], ax=ax).ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

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
    
    def calc_classes_based_position(self, classDistance: float = 0.1,
                                    positionX: np.ndarray = None,
                                    phaseTheta: np.ndarray = None,
                                    lookIdx: int = -1, 
                                    withPhase: bool = False) -> Tuple[List[List[int]], np.ndarray]:
        positionX, phaseTheta = self.check_state_input(positionX, phaseTheta, lookIdx)

        if withPhase:
            adjPositionX = np.concatenate([
                positionX, 
                phaseTheta.reshape(-1, 1) / (np.pi * 2) * self.model.boundaryLength
            ], axis=1)
        else:
            adjPositionX = positionX

        deltaX = self.model._delta_x(
            adjPositionX, adjPositionX[:, np.newaxis], 
            self.model.boundaryLength, self.model.halfBoundaryLength
        )
        totalDistances = (deltaX ** 2).sum(axis=-1) ** 0.5

        classes = self._calc_classes(adjPositionX, classDistance, totalDistances)
        return classes, positionX
    
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
    
    def calc_replative_distance(self, position1: np.ndarray, position2: np.ndarray):  #  -> float | np.ndarray
        deltaX = self.model._delta_x(position1, position2, 
                                     self.model.boundaryLength, 
                                     self.model.halfBoundaryLength)
        return np.linalg.norm(deltaX, axis=-1)

    def calc_abslute_distance(self, position1: np.ndarray, position2: np.ndarray) -> float:
        deltaX = position1 - position2
        return np.linalg.norm(deltaX, axis=-1)

    def calc_nearby_edges(self, classCenters: np.ndarray,
                          stdMulti: float = 0.3, 
                          relativeDistance: bool = False) -> Tuple[List[Tuple[int, int]], np.ndarray]:

        # For plot in periodic boundary conditions
        # classCenters be adjusted to include periodic images
        rawClassNums = classCenters.shape[0]
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

        tri = Delaunay(classCenters)
        edges = set()
        
        # get all edges from the Delaunay triangulation
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                edges.add(edge)
        # calculate the lengths of all edges
        edgeLengths = []
        for edge in edges:
            p1 = classCenters[edge[0]]
            p2 = classCenters[edge[1]]
            length = np.linalg.norm(p1 - p2)
            edgeLengths.append(length)
        # calculate mean and std of edge lengths
        meanLength = np.mean(edgeLengths)
        stdLength = np.std(edgeLengths)
        # filter edges based on the mean and std
        filteredEdges = []
        for i, edge in enumerate(edges):
            p1 = classCenters[edge[0]]
            p2 = classCenters[edge[1]]
            length = edgeLengths[i]

            if length <= meanLength + stdMulti * stdLength:
                filteredEdges.append(edge)
    
        if relativeDistance:
            edge = np.unique(np.mod(filteredEdges, rawClassNums), axis=0)
            return [tuple(edge) for edge in edge]
        else:
            return [tuple(edge) for edge in filteredEdges], classCenters

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
    
    def calc_order_parameter_Ra(self, phaseTheta: np.ndarray = None,
                                    lookIdx: int = -1, ajdDistance: float = 1) -> float:
        if phaseTheta is None:
            positionX, phaseTheta = self.get_state(lookIdx)

        adjacent = (
            self.calc_replative_distance(positionX, positionX[:, np.newaxis])
            <= ajdDistance
        )
        Rs = np.zeros(phaseTheta.shape[0])
        for i in range(phaseTheta.shape[0]):
            if not np.any(adjacent[i]):
                continue
            nearbyPhases = phaseTheta[adjacent[i]]
            Rs[i] = np.abs(np.mean(np.exp(1j * nearbyPhases)))

        return Rs.mean()


def calc_lattice_constants(sa: StateAnalysis, plot: bool = False, lookIdx: int = -1):

    sa: StateAnalysis
    model = sa.model
    shift = np.array([0., 0.])
    analysisRadius = model.speedV / np.abs(model.strengthK * np.sin(model.phaseLagA0))

    classes, centers = sa.calc_classes_and_centers(classDistance=analysisRadius, lookIdx=lookIdx)
    if len(classes) > model.agentsNum * 0.2:
        # print(f"Too many classes: {len(classes)} > {model.agentsNum * 0.2}, skipping.")
        return [], []
    numInClasses = np.array([len(c) for c in classes])
    # zScoreNum = stats.zscore(numInClasses)
    # classes = [classes[c] for c in range(len(classes)) 
    #            if (zScoreNum[c] > -0.4) and (numInClasses[c] > 10)]
    numThres = np.median(numInClasses[numInClasses > 10]) * 0.
    classes = [classes[c] for c in range(len(classes))
               if (numInClasses[c] > max(numThres, 10))]
    centers = np.mod(centers + shift, model.boundaryLength)
    if len(classes) <= 1:
        # print("Not enough classes, skipping.")
        return [], []

    classCenters: List[np.ndarray] = []
    for c in classes:
        singleClassCenters = centers[c]

        maxDeltaX = np.abs(singleClassCenters[:, 0] - singleClassCenters[:, 0, np.newaxis]).max()
        subXShift = model.halfBoundaryLength if maxDeltaX > model.halfBoundaryLength else 0
        maxDeltaY = np.abs(singleClassCenters[:, 1] - singleClassCenters[:, 1, np.newaxis]).max()
        subYShift = model.halfBoundaryLength if maxDeltaY > model.halfBoundaryLength else 0

        singleClassCenters = np.mod(singleClassCenters - np.array([subXShift, subYShift]), model.boundaryLength)
        classCenter = np.mod(singleClassCenters.mean(axis=0) + np.array([subXShift, subYShift]), model.boundaryLength)
        classCenters.append(classCenter)
    classCenters: np.ndarray = np.array(classCenters)

    edges, ajdClassCenters = sa.calc_nearby_edges(
        classCenters=classCenters, 
        stdMulti=0.3,
        relativeDistance=False
    )

    classAnalRadius = list()

    for _, oscIdx in enumerate(classes):
        freqOmega: np.ndarray = sa.model.freqOmega[oscIdx]
        meanFreq = freqOmega.mean()
        analRadius = model.speedV / np.abs(meanFreq - model.strengthK * np.sin(model.phaseLagA0))
        
        classAnalRadius.append(analRadius)

    classAnalRadius = np.array(classAnalRadius)
    edgeDistances = np.array([
        sa.calc_replative_distance(ajdClassCenters[edge[0]], ajdClassCenters[edge[1]]) 
        for edge in edges
    ])

    if plot:
        sa.plot_spatial(colorsBy="phase", index=-1, shift=shift)
        plt.scatter(
            classCenters[:, 0], classCenters[:, 1],
            facecolor="white", s=30, edgecolor="black", lw=1.5
        )
        for edge in edges:
            plt.plot(ajdClassCenters[edge, 0], ajdClassCenters[edge, 1],
                    color="black", lw=1.2, alpha=0.3, linestyle=(0, (10, 2)), zorder=0)

    # print(len(classes))

    return classAnalRadius, edgeDistances