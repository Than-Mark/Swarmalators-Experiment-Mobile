import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import sys
import os
import shutil

randomSeed = 100

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

import seaborn as sns

sns.set_theme(font_scale=1.1, rc={
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
plt.rcParams['mathtext.fontset'] = 'cm'

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


import numpy as np


class Solvable2DWithRepulsion(Swarmalators2D):
    def __init__(self, agentsNum: int, dt: float, 
                 K: float, J: float, P: float,
                 randomSeed: int = 100, tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, overWrite: bool = False) -> None:
        super().__init__(agentsNum, dt, K, randomSeed, tqdm, savePath, shotsnaps, overWrite)
        self.positionX = np.random.random((agentsNum, 2)) * 2 * np.pi
        self.J = J
        self.P = P
        self.one = np.ones((agentsNum, agentsNum))
        self.randomSeed = randomSeed

    def update_temp(self):
        self.temp["deltaTheta"] = self.deltaTheta
        self.temp["deltaX"] = self.deltaX

    def cotDeltaX(self, deltaX: np.ndarray) -> np.ndarray:
        """Cotangent of spatial difference: cot(x_j - x_i)"""
        return 1 / (np.tan(deltaX + (deltaX == 0)))

    @property
    def omega(self) -> np.ndarray:
        """Natural frequency: 0"""
        return 0

    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity: 0"""
        return 0

    @property 
    def Fatt(self) -> np.ndarray:
        """
        Effect of phase similarity on spatial attraction: J * cos(theta_j - theta_i) + 1
        """
        return self.J * np.cos(self.temp["deltaTheta"]) + 1
    
    @property
    def Frep(self) -> np.ndarray:
        """Effect of phase similarity on spatial repulsion: 1"""
        return self.one
    
    @property
    def Iatt(self) -> np.ndarray:
        """Spatial attraction: sin(x_j - x_i)"""
        return np.sin(self.temp["deltaX"])

    @property
    def Irep(self) -> np.ndarray:
        """Spatial repulsion: P * sin((x_j - x_i) / 2)"""
        return self.P * np.sin(2 * self.temp["deltaX"])

    @property
    def H(self) -> np.ndarray:
        """Phase interaction: sin(theta_j - theta_i)"""
        return np.sin(self.deltaTheta)
    
    @property
    def G(self) -> np.ndarray:
        """
        Effect of spatial similarity on phase couplings: cos(x_j - x_i) + cos(y_j - y_i)
        """
        return np.cos(self.deltaX).sum(axis=-1)

    @staticmethod
    @nb.njit
    def _update(
        positionX: np.ndarray, phaseTheta: np.ndarray,
        velocity: np.ndarray, omega: np.ndarray,
        Iatt: np.ndarray, Irep: np.ndarray,
        Fatt: np.ndarray, Frep: np.ndarray,
        H: np.ndarray, G: np.ndarray,
        K: float, dt: float
    ):
        dim = positionX.shape[0]
        pointX = velocity + np.sum(
            Iatt * Fatt.reshape((dim, dim, 1)) - Irep * Frep.reshape((dim, dim, 1)),
            axis=1
        ) / dim
        pointTheta = omega + K * np.sum(H * G, axis=1) / dim
        positionX = np.mod(positionX + pointX * dt, 2 * np.pi)
        phaseTheta = np.mod(phaseTheta + pointTheta * dt, 2 * np.pi)
        return positionX, phaseTheta

    def update(self) -> None:
        self.update_temp()
        self.positionX, self.phaseTheta = self._update(
            self.positionX, self.phaseTheta,
            self.velocity, self.omega,
            self.Iatt, self.Irep,
            self.Fatt, self.Frep,
            self.H, self.G,
            self.K, self.dt
        )
        self.counts += 1

    def plot(self, ax: plt.Axes = None, fixLim: bool = True) -> None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        sc = ax.scatter(self.positionX[:, 0], self.positionX[:, 1],
                    c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)

        cbar = plt.colorbar(sc, ticks=[0, np.pi, 2*np.pi], ax=ax)
        cbar.ax.set_ylim(0, 2*np.pi)
        cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])
        if fixLim:
            ax.set_xlim(0, 2*np.pi)
            ax.set_xticks([0, np.pi, 2*np.pi])
            ax.set_xticklabels(['$0$', '$\pi$', '$2\pi$'])
            ax.set_ylim(0, 2*np.pi)
            ax.set_yticks([0, np.pi, 2*np.pi])
            ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])

    def __str__(self):
        name = (
            f"Solvable2DWithRepulsion_"
            f"Agents.{self.agentsNum}_"
            f"K.{self.K}_"
            f"J.{self.J}_"
            f"P.{self.P}_"
            f"dt.{self.dt}_"
            f"seed.{self.randomSeed}"
        )
        return name
    

class StateAnalysis:
    def __init__(self, model: Solvable2DWithRepulsion = None, lookIndex: int = -1, showTqdm: bool = False):
        self.lookIndex = lookIndex
        self.showTqdm = showTqdm

        if model is not None:
            self.model = model
            targetPath = f"{self.model.savePath}/{self.model}.h5"
            totalPositionX = pd.read_hdf(targetPath, key="positionX")
            totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
            # totalPointX = pd.read_hdf(targetPath, key="pointX")
            # totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
            # totalDrivePosAndPhs = pd.read_hdf(targetPath, key="drivePosAndPhs")
            
            TNum = totalPositionX.shape[0] // self.model.agentsNum
            self.TNum = TNum
            self.tRange = np.arange(0, (TNum - 1) * model.shotsnaps, model.shotsnaps) * self.model.dt
            self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
            self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)
            # self.totalPointX = totalPointX.values.reshape(TNum, self.model.agentsNum, 2)
            # self.totalPointTheta = totalPointTheta.values.reshape(TNum, self.model.agentsNum)
            # totalDrivePosAndPhs = totalDrivePosAndPhs.values.reshape(TNum, 3)
            # self.totalDrivePosition = totalDrivePosAndPhs[:, :2]
            # self.totalDrivePhaseTheta = totalDrivePosAndPhs[:, 2]

            if self.showTqdm:
                self.iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
            else:
                self.iterObject = range(1, self.totalPhaseTheta.shape[0])

    def get_state(self, index: int = -1):
        return self.totalPositionX[index], self.totalPhaseTheta[index]

    @staticmethod
    def calc_order_parameter_R(model: Solvable2DWithRepulsion) -> float:
        return np.abs(np.sum(np.exp(1j * model.phaseTheta))) / model.agentsNum
    
    @staticmethod
    def calc_order_parameter_S(model: Solvable2DWithRepulsion) -> float:
        phi = np.arctan2(model.positionX[:, 1], model.positionX[:, 0])
        Sadd = np.abs(np.sum(np.exp(1j * (phi + model.phaseTheta)))) / model.agentsNum
        Ssub = np.abs(np.sum(np.exp(1j * (phi - model.phaseTheta)))) / model.agentsNum
        return np.max([Sadd, Ssub])

    @staticmethod
    def calc_order_parameter_Vp(model: Solvable2DWithRepulsion) -> float:
        pointX = model.temp["pointX"]
        phi = np.arctan2(pointX[:, 1], pointX[:, 0])
        return np.abs(np.sum(np.exp(1j * phi))) / model.agentsNum
    
    @staticmethod
    def calc_order_parameter_Ptr(model: Solvable2DWithRepulsion) -> float:
        pointTheta = model.temp["pointTheta"]
        Ntr = np.abs(pointTheta - model.driveThateVelocityOmega) < 0.2 / model.dt * 0.1
        return Ntr.sum() / model.agentsNum
    
    def plot_last_state(self, model: Solvable2DWithRepulsion = None, ax: plt.Axes = None, withColorBar: bool =True, 
                        s: float = 50, driveS: float = 100) -> None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        if model is not None:
            sc = ax.scatter(model.positionX[:, 0], model.positionX[:, 1], s=s,
                            c=model.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
            maxPos = np.abs(model.positionX).max()
        else:
            sc = ax.scatter(self.totalPositionX[self.lookIndex, :, 0], self.totalPositionX[self.lookIndex, :, 1], s=s,
                            c=self.totalPhaseTheta[self.lookIndex], cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
            maxPos = np.abs(self.totalPositionX[self.lookIndex]).max()
            
        # ax.set_xlim(0, 2*np.pi)
        # ax.set_xticks([0, np.pi, 2*np.pi])
        # ax.set_xticklabels(['$0$', '$\pi$', '$2\pi$'])
        # ax.set_ylim(0, 2*np.pi)
        # ax.set_yticks([0, np.pi, 2*np.pi])
        # ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])
        
        if withColorBar:
            cbar = plt.colorbar(sc, ticks=[0, np.pi, 2*np.pi], ax=ax)
            cbar.ax.set_ylim(0, 2*np.pi)
            cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])