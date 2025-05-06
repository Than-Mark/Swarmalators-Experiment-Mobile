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

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

randomSeed = 10

if "ipykernel_launcher.py" in sys.argv[0]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

colors = ["#403990", "#80A6E2", "#FBDD85", "#F46F43", "#CF3D3E"]
cmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


class ChiralActiveMatter(Swarmalators2D):
    def __init__(self, strengthLambda: float, distanceD0: float, 
                 boundaryLength: float = 10, speedV: float = 1.0,
                 agentsNum: int=1000, dt: float=0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, 
                 distribution: str = "uniform", randomSeed: int = 10, overWrite: bool = False) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.speedV = speedV
        self.distanceD0 = distanceD0
        if distribution == "uniform":
            self.omegaTheta = np.concatenate([
                np.random.uniform(1, 3, size=agentsNum // 2),
                np.random.uniform(-3, -1, size=agentsNum // 2)
            ])
        elif distribution == "normal":
            self.omegaTheta = np.concatenate([
                np.random.normal(loc=3, scale=0.5, size=agentsNum // 2),
                np.random.normal(loc=-3, scale=0.5, size=agentsNum // 2)
            ])
        elif distribution == "lorentzian":
            self.omegaTheta = np.concatenate([
                np.random.standard_cauchy(size=agentsNum // 2) + 3,
                np.random.standard_cauchy(size=agentsNum // 2) - 3
            ])
        if agentsNum == 2:
            self.omegaTheta = np.array([3, -3])

        self.distribution = distribution
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

    def plot(self, ax: plt.Axes = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
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
        return self._pointTheta(self.phaseTheta, self.omegaTheta, self.strengthLambda, self.dt, self.K)

    @staticmethod
    @nb.njit
    def _pointTheta(phaseTheta: np.ndarray, omegaTheta: np.ndarray, strengthLambda: float, 
                    h: float, K: np.ndarray):
        adjMatrixTheta = np.repeat(phaseTheta, phaseTheta.shape[0]).reshape(phaseTheta.shape[0], phaseTheta.shape[0])
        k1 = omegaTheta + strengthLambda * np.sum(K * np.sin(
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
        self.positionX = np.mod(self.positionX, self.boundaryLength)
        self.temp = self.pointTheta
        self.phaseTheta += self.temp
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi

    def __str__(self) -> str:
        
        name =  f"ChiralActiveMatter_{self.distribution}_{self.strengthLambda:.3f}_{self.distanceD0:.2f}_{self.randomSeed}"

        return name

    def close(self):
        if self.store is not None:
            self.store.close()


class ChiralActiveMatterNonreciprocalReact(ChiralActiveMatter):
    def __init__(self, strengthLambda: float, 
                 distanceD1Mean: float, distanceD2Mean: float, 
                 distanceD1Std: float = 0, distanceD2Std: float = 0,
                 chiralNum: int = 1, agentsNum: int=1000, dt: float=0.01, 
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, 
                 d0Distribution: str = "uniform", omegaDistribution: str = "uniform",
                 randomSeed: int = 10, overWrite: bool = False) -> None:
        super().__init__(strengthLambda, 0, 10, 3.0,
                         agentsNum, dt, tqdm, savePath, shotsnaps, omegaDistribution, randomSeed, overWrite)
        
        assert chiralNum in [1, 2], "chiralNum must be 1 or 2"
        assert d0Distribution in ["uniform", "normal", "lorentzian"], "d0Distribution must be uniform, normal or lorentzian"
        self.chiralNum = chiralNum
        self.strengthLambda = strengthLambda
        self.distanceD1Mean = distanceD1Mean
        self.distanceD2Mean = distanceD2Mean
        self.distanceD1Std = distanceD1Std
        self.distanceD2Std = distanceD2Std
        self.d0Distribution = d0Distribution
        self.omegaDistribution = omegaDistribution

        if chiralNum == 1:
            if d0Distribution == "uniform":
                self.distanceD0 = np.random.uniform(distanceD1Mean - distanceD1Std / 2, distanceD1Mean + distanceD1Std / 2, size=agentsNum)
                assert distanceD1Mean - distanceD1Std / 2 >= 0, "distanceD1Mean - distanceD1Std / 2 must be greater than 0"
            elif d0Distribution == "normal":
                self.distanceD0 = np.abs(np.random.normal(loc=distanceD1Mean, scale=distanceD1Std, size=agentsNum))
            elif d0Distribution == "lorentzian":
                self.distanceD0 = np.abs(np.random.standard_cauchy(size=agentsNum) * distanceD1Std + 3)  #半宽
        else:  # chiralNum == 2
            if d0Distribution == "uniform":
                self.distanceD0 = np.concatenate([
                    np.random.uniform(distanceD1Mean - distanceD1Std / 2, distanceD1Mean + distanceD1Std / 2, size=agentsNum // 2),
                    np.random.uniform(distanceD2Mean - distanceD2Std / 2, distanceD2Mean + distanceD2Std / 2, size=agentsNum // 2)
                ])
                assert distanceD1Mean - distanceD1Std / 2 >= 0, "distanceD1Mean - distanceD1Std / 2 must be greater than 0"
            elif d0Distribution == "normal":
                self.distanceD0 = np.concatenate([
                    np.abs(np.random.normal(loc=distanceD1Mean, scale=distanceD1Std, size=agentsNum // 2)),
                    np.abs(np.random.normal(loc=distanceD2Mean, scale=distanceD2Std, size=agentsNum // 2))
                ])
            elif d0Distribution == "lorentzian":
                self.distanceD0 = np.concatenate([
                    np.abs(np.random.standard_cauchy(size=agentsNum // 2) * distanceD1Std + 3),
                    np.abs(np.random.standard_cauchy(size=agentsNum // 2) * distanceD2Std + 3)
                ])

        singleOmegaDistributionSize = agentsNum if chiralNum == 1 else agentsNum // 2
        if omegaDistribution == "uniform":
            self.omegaTheta = np.random.uniform(1, 3, size=singleOmegaDistributionSize)
        elif omegaDistribution == "normal":
            self.omegaTheta = np.random.normal(loc=3, scale=0.5, size=singleOmegaDistributionSize)
        elif omegaDistribution == "lorentzian":
            self.omegaTheta = np.random.standard_cauchy(size=singleOmegaDistributionSize) + 3

        if chiralNum == 2:
            self.omegaTheta = np.concatenate([self.omegaTheta, -self.omegaTheta])

    def plot(self, fig: plt.Figure = None, axs: List[plt.Axes] = None):
        scale = 23
        width = 0.005
        
        text_content = (
            f'Singlechiral   '
            rf'$\lambda$: {self.strengthLambda:.2f}   '
            f'$d_0\sim$U   '
            f'D1: {self.distanceD1Mean:.2f}, D2: {self.distanceD2Mean:.2f}   '
            f'D1Std: {self.distanceD1Std:.2f}, D2Std: {self.distanceD2Std:.2f}   '
            rf'$\omega\sim$U(1,3)'
        )
        
        if self.distanceD1Std > 0:
            if fig is None:
                fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(12, 5.5))
                ax1, ax2 = axs
                plt.suptitle(text_content, fontweight='normal')
            else:
                ax1, ax2 = axs[0], axs[1]
            colors = [cmap(i) for i in (self.distanceD0 - self.distanceD1Mean) / self.distanceD1Std + 0.5]
            d0Min = np.round(self.distanceD0.min(), 2)
            d0Max = np.round(self.distanceD0.max(), 2)
            sca = ax1.scatter(
                -np.ones_like(self.omegaTheta), -np.ones_like(self.omegaTheta), 
                c=np.linspace(d0Min, d0Max, len(self.omegaTheta)), cmap=cmap, s=100)
            cbar = plt.colorbar(sca, ticks=[d0Min, np.round((d0Min + d0Max) / 2, 2), d0Max], ax=ax1)
            cbar.set_label(r"$d_0$", fontsize=16, rotation=0)
        else:
            if fig is None:
                fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 6]}, figsize=(11, 5))
                ax1, ax2 = axs
                plt.suptitle(text_content, fontweight='normal')
            else:
                ax1, ax2 = axs[0], axs[1]
            colors = [cmap(0) for _ in range(self.omegaTheta.size)]
        
        ax1.quiver(
            self.positionX[:, 0], self.positionX[:, 1],
            np.cos(self.phaseTheta[:]), np.sin(self.phaseTheta[:]), 
            color=colors, scale=scale, width=width
        )

        if self.chiralNum == 1:
            ax2.quiver(
                self.positionX[:, 0], self.positionX[:, 1],
                np.cos(self.phaseTheta[:]), np.sin(self.phaseTheta[:]), 
                color=[cmap(i) for i in (self.omegaTheta - 1) / 2],
                scale=scale, width=width
            )
            sca = ax2.scatter(
                -np.ones_like(self.omegaTheta), -np.ones_like(self.omegaTheta), 
                c=np.linspace(1, 3, len(self.omegaTheta)), cmap=cmap, s=100
            )
            cbar = plt.colorbar(sca, ticks=[1, 2, 3], ax=ax2)
            cbar.set_label(r"$\omega$", fontsize=16, rotation=0)
        elif self.chiralNum == 2:
            alphaRate = 0.8
            ax2.quiver(
                self.positionX[:self.agentsNum // 2, 0], self.positionX[:self.agentsNum // 2, 1],
                np.cos(self.phaseTheta[:self.agentsNum // 2]), np.sin(self.phaseTheta[:self.agentsNum // 2]), color='red', 
                alpha = (1 - alphaRate) + (np.abs(self.omegaTheta[:self.agentsNum // 2]) - 1) / 2 * alphaRate,
                scale=scale, width=width
            )
            ax2.quiver(
                self.positionX[self.agentsNum // 2:, 0], self.positionX[self.agentsNum // 2:, 1],
                np.cos(self.phaseTheta[self.agentsNum // 2:]), np.sin(self.phaseTheta[self.agentsNum // 2:]), color='#414CC7', 
                alpha = (1 - alphaRate) + (np.abs(self.omegaTheta[self.agentsNum // 2:]) - 1) / 2 * alphaRate,
                scale=scale, width=width
            )
            customCmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", ["#414CC7", "white", "red"])
            sca = ax2.scatter(
                -np.ones_like(self.omegaTheta), -np.ones_like(self.omegaTheta), 
                c=np.linspace(-3, 3, len(self.omegaTheta)), cmap=customCmap, s=100
            )
            cbar = plt.colorbar(sca, ticks=[-3, 0, 3], ax=ax2)
            cbar.set_label(r"$\omega$", fontsize=16, rotation=0)

        ax1.tick_params(direction='in', length=3)
        ax2.tick_params(direction='in', length=3)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)

        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.tight_layout()

        # plt.savefig(f"./mp4/{model}.mp4", dpi=100)
        # plt.close(fig)

    def __str__(self) -> str:
        
        name =  (
            f"ChiralActiveMatterNonreciprocalReact"
            f"_{self.chiralNum}"
            f"_{self.d0Distribution}"
            f"_{self.omegaDistribution}"
            f"_{self.strengthLambda:.3f}"
            f"_{self.distanceD1Mean:.2f}_{self.distanceD1Std:.2f}"
            f"_{self.distanceD2Mean:.2f}_{self.distanceD2Std:.2f}"
            f"_{self.randomSeed}"
        )
        
        return name


class StateAnalysis:
    def __init__(self, model: ChiralActiveMatter, classDistance: float = 2, lookIndex: int = -1, showTqdm: bool = False):
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
        return self._clac_phase_sync_op(theta)
    
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

def plot_drpaop(models: List[ChiralActiveMatter], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        drpa = sa.tv_dis_reciwgt_phase_agg_op(20)
        ax1.plot(drpa[:, 0], drpa[:, 1])
        ax1.set_ylim(0, 1)
        ax1.set_title(f"{model},     t: 0-12000, Dis ReciWgt Phase Agg")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_drpa(models: List[ChiralActiveMatter], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        drpa, colors = sa.tv_dis_reciwgt_phase_agg(30)
        ax1.scatter(drpa[:, 0], drpa[:, 1], s=0.5, alpha=0.01, c=colors)
        ax1.set_ylim(0, 1)
        ax1.set_title(f"{model},     t: 0-12000, Dis ReciWgt Phase Agg")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_ccop(models: List[ChiralActiveMatter], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        ccop = sa.tv_class_count_op(20)
        ax1.plot(ccop[:, 0], ccop[:, 1])
        ax1.set_ylim(0, 40)
        ax1.set_title(f"{model},     t: 0-12000, Class Count")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_cncop(models: List[ChiralActiveMatter], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        cncop = sa.tv_center_nearby_count_op(20)
        ax1.plot(cncop[:, 0], cncop[:, 1])
        ax1.set_ylim(0, 1)
        ax1.set_title(f"{model},     t: 0-12000, Center Nearby Count")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_cnc(models: List[ChiralActiveMatter], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        cnc, colors = sa.tv_center_nearby_count(30)
        ax1.scatter(cnc[:, 0], cnc[:, 1], s=0.5, alpha=0.01, c=colors)
        ax1.set_ylim(0, 1)
        ax1.set_title(f"{model},     t: 0-12000, Center Nearby Count")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_ldpaop(models: List[ChiralActiveMatter], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        ldpa = sa.tv_limit_ds_phase_agg_op(20)
        ax1.plot(ldpa[:, 0], ldpa[:, 1])
        ax1.set_ylim(0, 1)
        ax1.set_title(f"{model},     t: 0-12000, Limit Dis Phase Agg")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_ldpa(models: List[ChiralActiveMatter], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        ldpa, colors = sa.tv_limit_ds_phase_agg(30)
        ax1.scatter(ldpa[:, 0], ldpa[:, 1], s=0.5, alpha=0.01, c=colors)
        ax1.set_ylim(0, 1)
        ax1.set_title(f"{model},     t: 0-12000, Limit Dis Phase Agg")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_tvcp(models: List[ChiralActiveMatter], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1Row1 = plt.subplot2grid((len(models) * 2, 3), (idx, 0), colspan=2)
        ax1Row2 = plt.subplot2grid((len(models) * 2, 3), (idx + 1, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models) * 2, 3), (idx, 2), rowspan=2)

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        cp1, colors = sa.tv_center_position(step=30)
        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(rf"$\lambda={model.strengthLambda},\ d_0={model.distanceD0}$")
        cp1[:, 0] = cp1[:, 0]

        ax1Row1.scatter(cp1[:, 0], cp1[:, 1], s=0.5, alpha=0.01, c=colors)
        ax1Row1.set_ylim(0, 10)
        # ax1Row1.set_title(f"{model}\t, t: 0-12000, PositionX")
        ax1Row1.set_xlabel(f"$t$")
        ax1Row1.set_ylabel(f"$x$")
        ax1Row2.scatter(cp1[:, 0], cp1[:, 2], s=0.5, alpha=0.01, c=colors)
        ax1Row2.set_ylim(0, 10)
        ax1Row2.set_xlabel(f"$t$")
        ax1Row2.set_ylabel(f"$y$")

        idx += 2

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_tvca(models: List[ChiralActiveMatter], opType: int, savePath: str = None):
    assert opType in [1, 2], "opType must be 1 or 2"
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        ca, colors = sa.tv_center_agg(opType, 30)
        ax1.scatter(ca[:, 0], ca[:, 1], s=0.5, alpha=0.01, c=colors)
        ax1.set_title(f"{model},     t: 0-12000, Center Agg{opType}")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()


def plot_tvcaop(models: List[ChiralActiveMatter], opType: int, savePath: str = None):
    assert opType in [1, 2], "opType must be 1 or 2"
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        caop = sa.tv_center_agg_op(opType, 20)
        ax1.plot(caop[:, 0], caop[:, 1])
        ax1.set_title(f"{model},     t: 0-12000, Center Agg Order Parameter{opType}")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()

def plot_tvcr(models: List[ChiralActiveMatter], savePath: str = None):
    _ = plt.figure(figsize=(3 * 5, len(models) * 5))

    idx = 0

    for model in tqdm(models):
        ax1 = plt.subplot2grid((len(models), 3), (idx, 0), colspan=2)
        ax2 = plt.subplot2grid((len(models), 3), (idx, 2))

        sa = StateAnalysis(model, classDistance=1, lookIndex=-1, tqdm=False)
        cr, colors = sa.tv_center_radius(30)
        ax1.scatter(cr[:, 0], cr[:, 1], s=0.5, alpha=0.01, c=colors)
        ax1.set_title(f"{model},     t: 0-12000, Global center distance")

        sa.plot_centers(ax=ax2, index=-1)
        ax2.set_title(f"snapshot at 12000")

        idx += 1

    plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=100, bbox_inches="tight")
    plt.close()


def draw_mp4(model: ChiralActiveMatterNonreciprocalReact):

    targetPath = f"./data/{model}.h5"
    totalPositionX = pd.read_hdf(targetPath, key="positionX")
    totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
    totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
    TNum = totalPositionX.shape[0] // model.agentsNum
    totalPositionX = totalPositionX.values.reshape(TNum, model.agentsNum, 2)
    totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, model.agentsNum)
    totalPointTheta = totalPointTheta.values.reshape(TNum, model.agentsNum)
    shift = 0
    class1, class2 = (
        np.concatenate([np.ones(model.agentsNum // 2), np.zeros(model.agentsNum // 2)]).astype(bool), 
        np.concatenate([np.zeros(model.agentsNum // 2), np.ones(model.agentsNum // 2)]).astype(bool)
    )

    def plot_frame(i):
        pbar.update(1)
        positionX = totalPositionX[i]
        phaseTheta = totalPhaseTheta[i]
        fig.clear()
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3, projection='3d')
        model.positionX = positionX
        model.phaseTheta = phaseTheta
        model.plot(fig=fig, axs=[ax1, ax2])

        text_content = (
            f'Singlechiral   '
            rf'\lambda: {model.strengthLambda:.2f}   '
            f'd_0\simU   '
            f'D1: {model.distanceD1Mean:.2f}, D2: {model.distanceD2Mean:.2f}   '
            f'D1Std: {model.distanceD1Std:.2f}, D2Std: {model.distanceD2Std:.2f}   '
            rf'\omega\simU(1,3)'
        )

        fig.text(0.5, 0.95, text_content, ha='center', va='center', fontsize=10, color='black')

        hist, bins = np.histogram(phaseTheta[class1], bins=100, range=(-np.pi, np.pi))

        ax3.plot_surface(
            np.cos(bins[:-1]), np.sin(bins[:-1]), 
            np.array([np.zeros_like(hist), hist]), 
            color='tomato', alpha=0.5, edgecolor="tomato"
        )
        hist, bins = np.histogram(phaseTheta[class2], bins=100, range=(-np.pi, np.pi))
        ax3.plot_surface(
            np.cos(bins[:-1]) + shift, np.sin(bins[:-1]) + shift,
            np.array([np.zeros_like(hist), hist]), 
            color='dodgerblue', alpha=0.5, edgecolor="dodgerblue"
        )
        ax3.set_xlabel(r"$\cos(\theta_I)$")
        ax3.set_ylabel(r"$\sin(\theta_I)$")
        ax3.set_zlabel("Count")

        plt.tight_layout()

    mp4TNum = totalPositionX.shape[0]
    pbar = tqdm(total=mp4TNum + 1)
    fig, axs = plt.subplots(figsize=(17, 5))
    
    ani = ma.FuncAnimation(fig, plot_frame, frames=np.arange(0, mp4TNum, 1), interval=50, repeat=False)

    ani.save(f"./mp4/{model}.mp4", dpi=100, writer="ffmpeg")

    plt.close()

    pbar.close()


def plot_last(model: ChiralActiveMatterNonreciprocalReact):

    targetPath = f"./data/{model}.h5"
    totalPositionX = pd.read_hdf(targetPath, key="positionX")
    totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
    totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
    TNum = totalPositionX.shape[0] // model.agentsNum
    totalPositionX = totalPositionX.values.reshape(TNum, model.agentsNum, 2)
    totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, model.agentsNum)
    totalPointTheta = totalPointTheta.values.reshape(TNum, model.agentsNum)

    positionX = totalPositionX[-1]
    phaseTheta = totalPhaseTheta[-1]
    model.positionX = positionX
    model.phaseTheta = phaseTheta

    # fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(12, 5))
    # ax1, ax2 = axs

    # fig = plt.figure()

    # model.plot(fig=fig, axs=[ax1, ax2])
    model.plot()
    plt.savefig(f"./fig/{model}.png", dpi=150)
    # plt.close()
    # plt.show()