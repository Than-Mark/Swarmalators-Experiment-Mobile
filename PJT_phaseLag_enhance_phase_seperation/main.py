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


if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


class ChiralInducedPhaseLag(Swarmalators2D):
    def __init__(self, strengthLambda: float, distanceD0: float, 
                 boundaryLength: float = 10, speedV: float = 3.0,
                 phaseLag: float = 0, 
                 omegaMin: float = 0.1, omegaMax: float = 3.0,
                 agentsNum: int=1000, dt: float=0.02, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, 
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.speedV = speedV
        self.distanceD0 = distanceD0
        self.omegaMin = omegaMin
        self.omegaMax = omegaMax
        posOmega = np.random.uniform(omegaMin, omegaMax, agentsNum // 2)
        self.omegaTheta = np.concatenate([
            posOmega, -posOmega      
        ])

        self.strengthLambda = strengthLambda
        self.tqdm = tqdm
        self.savePath = savePath
        self.temp = np.zeros(agentsNum)
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.boundaryLength = boundaryLength
        self.halfBoundaryLength = boundaryLength / 2
        self.randomSeed = randomSeed
        self.overWrite = overWrite

        self.phaseLag = phaseLag

        self.omegaTheta = np.sort(self.omegaTheta)
        isPos = (self.omegaTheta > 0).astype(np.int32)
        negNum = (self.omegaTheta < 0).sum()
        self.phaseLagMatrix = np.concatenate([
            np.repeat(isPos, negNum).reshape(self.agentsNum, negNum).T,
            np.repeat(1 - isPos, self.agentsNum - negNum).reshape(self.agentsNum, self.agentsNum - negNum).T
        ]) * phaseLag

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
        plt.xlim(0, 10)
        plt.ylim(0, 10)

    @property
    def K(self):
        return self.distance_x(self.deltaX) <= self.distanceD0

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
        return self._pointTheta(self.phaseTheta, self.omegaTheta, self.strengthLambda, 
                                self.K, self.phaseLagMatrix)

    @staticmethod
    @nb.njit
    def _pointTheta(phaseTheta: np.ndarray, omegaTheta: np.ndarray, strengthLambda: float, 
                    K: np.ndarray, phaseLagMatrix: np.ndarray) -> np.ndarray:
        adjMatrixTheta = np.repeat(phaseTheta, phaseTheta.shape[0]).reshape(phaseTheta.shape[0], phaseTheta.shape[0])
        return omegaTheta + strengthLambda * np.sum(K * (
            np.sin(adjMatrixTheta - phaseTheta + phaseLagMatrix) - np.sin(phaseLagMatrix)
        ), axis=0)

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
        self.positionX = np.mod(self.positionX, self.boundaryLength)
        self.temp = self.pointTheta * self.dt
        self.phaseTheta += self.temp
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi

    def __str__(self) -> str:
        
        return (
            f"ChiralInducedPhaseLag_"
            f"_{self.strengthLambda:.3f}_{self.distanceD0:.2f}"
            f"_{self.phaseLag:.2f}_{self.omegaMin:.2f}_{self.omegaMax:.2f}"
            f"_{self.randomSeed}"
        )

    def close(self):
        if self.store is not None:
            self.store.close()


class StateAnalysis:
    def __init__(self, model: ChiralInducedPhaseLag, classDistance: float = 2, lookIndex: int = -1, showTqdm: bool = False):
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
    
    # @staticmethod
    # @nb.njit
    # def _calc_centers(positionX, phaseTheta, pointTheta, speedV):
    #     centers = np.zeros((positionX.shape[0], 2))

    #     for i in range(positionX.shape[0]):
    #         counts = max(200, abs(2 * np.pi // pointTheta[i]))
    #         phaseChanges = np.arange(0, counts * pointTheta[i], pointTheta[i])

    #         trackX = positionX[i, 0] + np.cumsum(speedV * np.cos(phaseTheta[i] + phaseChanges))
    #         trackY = positionX[i, 1] + np.cumsum(speedV * np.sin(phaseTheta[i] + phaseChanges))

    #         centers[i, 0] = np.mean(trackX)  # np.sum(np.array([trackX, trackY]), axis=1) / counts
    #         centers[i, 1] = np.mean(trackY)  

    #     return centers

    @staticmethod
    @nb.njit
    def _calc_centers(positionX, phaseTheta, pointTheta, speedV, dt):
        centers = np.zeros((positionX.shape[0], 2))

        for i in range(positionX.shape[0]):
            position, phase, point = positionX[i], phaseTheta[i], pointTheta[i]
            point1 = position
            velocity1 = speedV * dt * np.array([np.cos(phase), np.sin(phase)])
            point2 = position + velocity1
            velocity2 = speedV * dt * np.array([np.cos(phase + point), np.sin(phase + point)])

            unit_velocity1 = velocity1 / np.linalg.norm(velocity1)
            unit_velocity2 = velocity2 / np.linalg.norm(velocity2)

            coefficients = np.array([
                [unit_velocity1[0], unit_velocity1[1]],
                [unit_velocity2[0], unit_velocity2[1]]
            ])

            constants = np.array([
                np.sum(unit_velocity1 * point1),
                np.sum(unit_velocity2 * point2)
            ])

            center = np.linalg.solve(coefficients, constants)

            centers[i] = center

        return centers

    @property
    def centers(self):
        
        lastPositionX, lastPhaseTheta, lastPointTheta = self.get_state(self.lookIndex)
        
        centers = self._calc_centers(
            lastPositionX, lastPhaseTheta, lastPointTheta, self.model.speedV, self.model.dt
        )

        return np.mod(centers, self.model.boundaryLength)

    @property
    def centersNoMod(self):
            
        lastPositionX, lastPhaseTheta, lastPointTheta = self.get_state(self.lookIndex)
        
        centers = self._calc_centers(
            lastPositionX, lastPhaseTheta, lastPointTheta, self.model.speedV, self.model.dt
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
        positionX, phaseTheta, pointTheta = self.get_state(index)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        if oscis is None:
            oscis = np.arange(self.model.agentsNum)

        ax.quiver(
            positionX[oscis, 0], positionX[oscis, 1],
            np.cos(phaseTheta[oscis]), np.sin(phaseTheta[oscis]), **kwargs
        )
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)    

    def plot_centers(self, ax: plt.Axes = None, index: int = -1):
        positionX, phaseTheta, pointTheta = self.get_state(index)

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
    
    def center_radius_op(self, classOsci: np.ndarray):
        centers = self.centers
        return self.adj_distance(centers[classOsci], self.totalPositionX[:, classOsci]).mean(axis=1)
    
    def tv_center_radius_op(self, step: int = 10):
        centerRadios = []

        if self.showTqdm:
            iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            iterObject = range(1, self.totalPhaseTheta.shape[0])

        for i in iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i
            centers = self.centersNoMod
            modCenters = np.mod(centers, self.model.boundaryLength)
            distance = self.adj_distance(modCenters, self.totalPositionX[i])
            d = np.mean(distance)
            centerRadios = centerRadios + [d] * step

        return np.array(centerRadios)
    
    def tv_center_position(self, step: int = 30):
        color = ["red"] * 500 + ["blue"] * 500

        t = []
        positionX = []
        positionY = []
        colors = []

        if self.showTqdm:
            iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            iterObject = range(1, self.totalPhaseTheta.shape[0])

        for i in iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            centers = self.centers
            t.append(np.ones(self.model.agentsNum) * i)
            positionX.append(centers[:, 0])
            positionY.append(centers[:, 1])
            colors.append(color)

        t = np.concatenate(t, axis=0)
        positionX = np.concatenate(positionX, axis=0)
        positionY = np.concatenate(positionY, axis=0)
        colors = np.concatenate(colors, axis=0)

        return np.array([t, positionX, positionY]).T, colors

    def tv_center_radius(self, step: int = 30):
        
        color = ["tomato"] * 500 + ["dodgerblue"] * 500

        t = []
        centerRadios = []
        colors = []

        if self.showTqdm:
            iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            iterObject = range(1, self.totalPhaseTheta.shape[0])

        for i in iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            centers = self.centersNoMod
            t.append(np.ones(self.model.agentsNum) * i)
            centerRadios.append(
                np.sqrt(np.sum((centers - self.totalPositionX[i])**2, axis=-1))
            )  # self.adj_distance(self.totalPositionX[i], centers)
            colors.append(color)

        t = np.concatenate(t, axis=0)
        centerRadios = np.concatenate(centerRadios, axis=0)
        colors = np.concatenate(colors, axis=0)

        return np.array([t, centerRadios]).T, colors
    
    def phase_agg_op(self):
        theta = self.totalPhaseTheta[self.lookIndex]
        return self._clac_phase_sync_op(self, theta)
    
    @staticmethod
    @nb.njit
    def _clac_phase_sync_op(theta):
        N = theta.shape[0]
        return (
            (np.sum(np.sin(theta)) / N) ** 2 + 
            (np.sum(np.cos(theta)) / N) ** 2
        )**0.5

    @staticmethod
    @nb.njit
    def _delta_x(positionX, others, boundaryLength, halfLength):
        subX = positionX - others
        adjustOthers = (
            others * (-halfLength <= subX) * (subX <= halfLength) + 
            (others - boundaryLength) * (subX < -halfLength) + 
            (others + boundaryLength) * (subX > halfLength)
        )
        adjustSubX = positionX - adjustOthers
        return adjustSubX

    def delta_x(self, positionX, others):
        return self._delta_x(
            positionX, others, self.model.boundaryLength, self.model.halfBoundaryLength
        )

    @property
    def center_agg1(self):

        centers = self.centers
        deltaX = self.delta_x(centers, centers[:, np.newaxis])

        return np.sqrt(np.sum(deltaX ** 2, axis=2)).mean(axis=1)

    @property
    def center_agg2(self):

        centers = self.centers
        deltaX = self.delta_x(centers, centers[:, np.newaxis])

        return np.sqrt(np.sum(deltaX.mean(axis=1) ** 2, axis=-1))
    
    @property
    def dis_reciwgt_phase_agg(self):
        centers = self.centers
        deltaX = self.delta_x(centers, centers[:, np.newaxis])
        distance = np.sqrt(np.sum(deltaX ** 2, axis=-1))
        centerDisMat = distance.copy()
        centerDisMat[centerDisMat != 0] = 1 / centerDisMat[centerDisMat != 0]
        N = centers.shape[0]
        theta = self.totalPhaseTheta[self.lookIndex]
        return [
            (
                (np.sum(np.sin(theta) * centerDisMat[rowIdx]) / centerDisMat[rowIdx].sum()) ** 2 + 
                (np.sum(np.cos(theta) * centerDisMat[rowIdx]) / centerDisMat[rowIdx].sum()) ** 2
            )**0.5
            for rowIdx in range(N)
        ]
    
    @property
    def limit_dis_phase_agg(self):
        centers = self.centers
        deltaX = self.delta_x(centers, centers[:, np.newaxis])
        distance = np.sqrt(np.sum(deltaX ** 2, axis=-1))
        centerDisMat = distance < self.classDistance
        N = centers.shape[0]
        theta = self.totalPhaseTheta[self.lookIndex]
        return [
            (
                (np.sum(np.sin(theta) * centerDisMat[rowIdx]) / centerDisMat[rowIdx].sum()) ** 2 + 
                (np.sum(np.cos(theta) * centerDisMat[rowIdx]) / centerDisMat[rowIdx].sum()) ** 2
            )**0.5
            for rowIdx in range(N)
        ]

    def tv_dis_reciwgt_phase_agg(self, step: int = 30):
        color = ["tomato"] * 500 + ["dodgerblue"] * 500

        t = []
        opValues = []
        colors = []

        for i in self.iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            opValue = self.dis_reciwgt_phase_agg

            t.append(np.ones(self.model.agentsNum) * i)
            opValues.append(opValue)
            colors.append(color)

        t = np.concatenate(t, axis=0)
        opValues = np.concatenate(opValues, axis=0)
        colors = np.concatenate(colors, axis=0)

        return np.array([t, opValues]).T, colors

    def tv_dis_reciwgt_phase_agg_op(self, step: int = 10):
        t = []
        r = []

        for i in self.iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            opValue = self.dis_reciwgt_phase_agg

            r.append(np.mean(opValue))
            t.append(i)

        return np.array([t, r]).T

    def tv_limit_ds_phase_agg(self, step: int = 30):
        color = ["tomato"] * 500 + ["dodgerblue"] * 500

        t = []
        opValues = []
        colors = []

        for i in self.iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            opValue = self.limit_dis_phase_agg

            t.append(np.ones(self.model.agentsNum) * i)
            opValues.append(opValue)
            colors.append(color)

        t = np.concatenate(t, axis=0)
        opValues = np.concatenate(opValues, axis=0)
        colors = np.concatenate(colors, axis=0)

        return np.array([t, opValues]).T, colors

    def tv_limit_ds_phase_agg_op(self, step: int = 10):
        t = []
        r = []

        for i in self.iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            opValue = self.limit_dis_phase_agg

            r.append(np.mean(opValue))
            t.append(i)

        return np.array([t, r]).T

    def tv_class_count_op(self, step: int = 30):
        t = []
        r = []

        for i in self.iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            classes, _ = self.get_classes_centers()
            r.append(len(classes))
            t.append(i)

        return np.array([t, r]).T

    def tv_center_nearby_count_op(self, step: int = 30):
        t = []
        r = []

        for i in self.iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            centers = self.centers
            deltaX = self.delta_x(centers, centers[:, np.newaxis])
            distance = np.sqrt(np.sum(deltaX ** 2, axis=-1))
            r.append(
                (distance < self.classDistance).sum(axis=1).mean() / (self.model.agentsNum // 2)
            )
            t.append(i)

        return np.array([t, r]).T

    def tv_center_nearby_count(self, step: int = 30):
        color = ["tomato"] * 500 + ["dodgerblue"] * 500

        t = []
        r = []
        colors = []

        for i in self.iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            centers = self.centers
            deltaX = self.delta_x(centers, centers[:, np.newaxis])
            distance = np.sqrt(np.sum(deltaX ** 2, axis=-1))
            r.append(
                (distance < self.classDistance).sum(axis=1) / (self.model.agentsNum // 2)
            )
            t.append(np.ones(self.model.agentsNum) * i)
            colors.append(color)

        t = np.concatenate(t, axis=0)
        r = np.concatenate(r, axis=0)
        colors = np.concatenate(colors, axis=0)

        return np.array([t, r]).T, colors

    def tv_center_agg(self, opType: int, step: int = 30):
        color = ["tomato"] * 500 + ["dodgerblue"] * 500

        t = []
        centerAggs = []
        colors = []

        if self.showTqdm:
            iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            iterObject = range(1, self.totalPhaseTheta.shape[0])

        for i in iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            if opType == 1:
                caValue = self.center_agg1
            elif opType == 2:
                caValue = self.center_agg2

            t.append(np.ones(self.model.agentsNum) * i)
            centerAggs.append(caValue)
            colors.append(color)

        t = np.concatenate(t, axis=0)
        centerAggs = np.concatenate(centerAggs, axis=0)
        colors = np.concatenate(colors, axis=0)

        return np.array([t, centerAggs]).T, colors
    
    def tv_center_agg_op(self, opType: int, step: int = 10):
        assert opType in [1, 2], "opType must be 1 or 2"

        t = []
        r = []

        if self.showTqdm:
            iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
        else:
            iterObject = range(1, self.totalPhaseTheta.shape[0])

        for i in iterObject:
            if i % step != 0:
                continue
            self.lookIndex = i

            if opType == 1:
                caValue = self.center_agg1
            elif opType == 2:
                caValue = self.center_agg2

            r.append(np.mean(caValue))
            t.append(i)

        return np.array([t, r]).T