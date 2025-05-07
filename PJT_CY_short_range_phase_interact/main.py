import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import product
import pandas as pd
import numpy as np
import numba as nb
import imageio
import sys
import os
import shutil

if "ipykernel_launcher.py" in sys.argv[0]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

randomSeed = 100

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
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

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D

import numpy as np


class ShortRangePhaseInter1(Swarmalators2D):
    def __init__(self, K: float, J: float, d0: float, 
                 agentsNum: int = 500, dt: float = 0.01, 
                 randomSeed: int = 100, tqdm: bool = False, savePath: str = None, shotsnaps: int = 100, overWrite: bool = False) -> None:
        super().__init__(agentsNum, dt, K, randomSeed, tqdm, savePath, shotsnaps, overWrite)
        self.J = J
        self.d0 = d0
        self.one = np.ones((agentsNum, agentsNum))
    
    @property
    def Fatt(self) -> np.ndarray:
        """
        Effect of phase similarity on spatial attraction: 1 + J * A * cos(theta_j - theta_i)
        """
        return 1 + self.J * self.A * np.cos(self.temp["deltaTheta"])

    @property
    def Frep(self) -> np.ndarray:
        """Effect of phase similarity on spatial repulsion: 1"""
        return 1

    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity: 0"""
        return 0
    
    @property
    def omega(self) -> np.ndarray:
        """Natural frequency: 0"""
        return 0
    
    @property
    def A(self) -> np.ndarray:
        """Adjacency matrix: 1 if |x_i - x_j| <= d0 else 0"""
        return np.where(self.temp["distanceX"] <= self.d0, 1, 0)

    def update(self) -> None:
        self.update_temp()
        self.positionX, self.phaseTheta = self._update(
            self.positionX, self.phaseTheta,
            self.dotPosition, self.dotTheta,
            self.dt
        )
        self.counts += 1

    @staticmethod
    @nb.njit
    def _update(
        positionX: np.ndarray, phaseTheta: np.ndarray,
        dotPosition: np.ndarray, dotTheta: np.ndarray,
        dt: float
    ):
        dotX, dotY = dotPosition
        positionX[:, 0] = positionX[:, 0] + dotX * dt
        positionX[:, 1] = positionX[:, 1] + dotY * dt
        phaseTheta = np.mod(phaseTheta + dotTheta * dt, 2 * np.pi)
        return positionX, phaseTheta

    def update_temp(self):
        self.temp["deltaTheta"] = self.deltaTheta
        self.temp["deltaX"] = self.deltaX
        self.temp["distanceX"] = self.distance_x(self.temp["deltaX"])
        self.temp["distanceX"] = np.where(self.temp["distanceX"] == 0, 1, self.temp["distanceX"])

    @property
    def dotPosition(self):
        return self.calc_dot_position(
            deltaX=self.temp["deltaX"][:, :, 0], 
            deltaY=self.temp["deltaX"][:, :, 1], 
            Fatt=self.Fatt,
            Frep=self.Frep,
            distance=self.temp["distanceX"]
        )

    @staticmethod
    @nb.njit
    def calc_dot_position(deltaX: np.ndarray, deltaY: np.ndarray, 
                          Fatt: np.ndarray, Frep: np.ndarray,
                          distance: np.ndarray):
        IattX = deltaX / distance
        IattY = deltaY / distance
        IrepX = IattX / distance
        IrepY = IattY / distance
        dotX = np.sum(IattX * Fatt - IrepX * Frep, axis=1) / distance.shape[0]
        dotY = np.sum(IattY * Fatt - IrepY * Frep, axis=1) / distance.shape[0]
        return dotX, dotY

    @property
    def dotTheta(self) -> np.ndarray:
        return self._calc_dot_theta(self.temp["deltaTheta"], self.A, self.omega, self.K)

    @staticmethod
    @nb.njit
    def _calc_dot_theta(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, K: float) -> np.ndarray:
        coupling = np.zeros(deltaTheta.shape[0])
        for idx in range(deltaTheta.shape[0]):
            coupling[idx] = np.mean(np.sin(deltaTheta[idx][A[idx] == 1]))
        return K * coupling + omega

    def plot(self, ax: plt.Axes = None) -> None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))


        sc = ax.scatter(self.positionX[:, 0], self.positionX[:, 1],
                    c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)

        cbar = plt.colorbar(sc, ticks=[0, np.pi, 2*np.pi], ax=ax)
        cbar.ax.set_ylim(0, 2*np.pi)
        cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])
        ax.set_title(rf"$J={self.J:.2f},\ K={self.K:.2f},\ d_0={self.d0}$")

    def __str__(self) -> str:
        return f"ShortRangePhaseInter1_a{self.agentsNum}_K{self.K:.2f}_J{self.J:.2f}_d0{self.d0:.2f}"
    

class ShortRangePhaseInter2(ShortRangePhaseInter1):
    @property
    def dotTheta(self) -> np.ndarray:
        return self._calc_dot_theta(self.temp["deltaTheta"], self.A, self.omega, self.K, self.temp["distanceX"])

    @staticmethod
    @nb.njit
    def _calc_dot_theta(deltaTheta: np.ndarray, A: np.ndarray, omega: np.ndarray, K: float, distance: np.ndarray) -> np.ndarray:
        coupling = np.zeros(deltaTheta.shape[0])
        for idx in range(deltaTheta.shape[0]):
            coupling[idx] = np.mean(
                np.sin(deltaTheta[idx][A[idx] == 1]) / distance[idx][A[idx] == 1]
            )
        return K * coupling + omega
    
    def __str__(self) -> str:
        return f"ShortRangePhaseInter2_a{self.agentsNum}_K{self.K:.2f}_J{self.J:.2f}_d0{self.d0:.2f}"


class StateAnalysis:
    def __init__(self, model: ShortRangePhaseInter1 = None, lookIndex: int = -1, showTqdm: bool = False):
        self.lookIndex = lookIndex
        self.showTqdm = showTqdm

        if model is not None:
            self.model = model
            targetPath = f"{self.model.savePath}/{self.model}.h5"
            totalPositionX = pd.read_hdf(targetPath, key="positionX")
            totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
            
            TNum = totalPositionX.shape[0] // self.model.agentsNum
            self.TNum = TNum
            self.tRange = np.arange(0, (TNum - 1) * model.shotsnaps, model.shotsnaps) * self.model.dt
            self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
            self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)

            if self.showTqdm:
                self.iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
            else:
                self.iterObject = range(1, self.totalPhaseTheta.shape[0])

    def get_state(self, index: int = -1):
        return self.totalPositionX[index], self.totalPhaseTheta[index]

    @staticmethod
    def calc_order_parameter_R(model: ShortRangePhaseInter1) -> float:
        return np.abs(np.sum(np.exp(1j * model.phaseTheta))) / model.agentsNum
    
    @staticmethod
    def calc_order_parameter_S(model: ShortRangePhaseInter1) -> float:
        phi = np.arctan2(model.positionX[:, 1], model.positionX[:, 0])
        Sadd = np.abs(np.sum(np.exp(1j * (phi + model.phaseTheta)))) / model.agentsNum
        Ssub = np.abs(np.sum(np.exp(1j * (phi - model.phaseTheta)))) / model.agentsNum
        return np.max([Sadd, Ssub])

    @staticmethod
    def calc_order_parameter_Vp(model: ShortRangePhaseInter1) -> float:
        pointX = model.temp["pointX"]
        phi = np.arctan2(pointX[:, 1], pointX[:, 0])
        return np.abs(np.sum(np.exp(1j * phi))) / model.agentsNum
    
    @staticmethod
    def calc_order_parameter_Ptr(model: ShortRangePhaseInter1) -> float:
        pointTheta = model.temp["pointTheta"]
        Ntr = np.abs(pointTheta - model.driveThateVelocityOmega) < 0.2 / model.dt * 0.1
        return Ntr.sum() / model.agentsNum
    
    def plot_last_state(self, model: ShortRangePhaseInter1 = None, ax: plt.Axes = None, withColorBar: bool =True, 
                        s: float = 50) -> None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        if model is not None:
            t = model.counts * model.dt
            sc = ax.scatter(model.positionX[:, 0], model.positionX[:, 1], s=s,
                            c=model.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
            maxPos = np.abs(model.positionX).max()
            ax.set_title(rf"$J={model.J:.2f},\ K={model.K:.2f},\ d_0={model.d0}$")
        else:
            sc = ax.scatter(self.totalPositionX[self.lookIndex, :, 0], self.totalPositionX[self.lookIndex, :, 1], s=s,
                            c=self.totalPhaseTheta[self.lookIndex], cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
            maxPos = np.abs(self.totalPositionX[self.lookIndex]).max()
            ax.set_title(rf"$J={self.model.J:.2f},\ K={self.model.K:.2f},\ d_0={self.model.d0}$")

        if maxPos < 1:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        else:
            bound = maxPos * 1.05
            roundBound = np.round(bound)
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)
            ax.set_xticks([-roundBound, -roundBound / 2, 0, roundBound / 2, roundBound])
            ax.set_yticks([-roundBound, -roundBound / 2, 0, roundBound / 2, roundBound])
        
        if withColorBar:
            cbar = plt.colorbar(sc, ticks=[0, np.pi, 2*np.pi], ax=ax)
            cbar.ax.set_ylim(0, 2*np.pi)
            cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])



        
    
    def plot_state_at_step(self, model: ShortRangePhaseInter1 = None, ax: plt.Axes = None, withColorBar: bool =True, 
                           step: int = 9000, s: float = 50) -> None:
        """
        绘制指定时间步 (step) 的模型状态，如果没有传递 step，则绘制最后一步的状态
        :param model: ShortRangePhaseInter1 模型
        :param ax: matplotlib 的坐标轴对象
        :param step: 要绘制的时间步，如果为 None,则绘制最后一步
        :param withColorBar: 是否显示颜色条
        :param s: 点的大小
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        if model is not None:
            t = model.counts * model.dt
            sc = ax.scatter(model.positionX[:, 0], model.positionX[:, 1], s=s,
                            c=model.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
            maxPos = np.abs(model.positionX).max()
            ax.set_title(rf"$J={model.J:.2f},\ K={model.K:.2f},\ d_0={model.d0}$")
        else:
            sc = ax.scatter(self.totalPositionX[step, :, 0], self.totalPositionX[step, :, 1], 
                            s=s, c=self.totalPhaseTheta[step], cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
            maxPos = np.abs(self.totalPositionX[step]).max()
            ax.set_title(rf"$J={self.model.J:.2f},\ K={self.model.K:.2f},\ d_0={self.model.d0}$")
        
        if maxPos < 1:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        else:
            bound = maxPos * 1.05
            roundBound = np.round(bound)
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)
            ax.set_xticks([-roundBound, -roundBound / 2, 0, roundBound / 2, roundBound])
            ax.set_yticks([-roundBound, -roundBound / 2, 0, roundBound / 2, roundBound])
        
        if withColorBar:
            cbar = plt.colorbar(sc, ticks=[0, np.pi, 2*np.pi], ax=ax)
            cbar.ax.set_ylim(0, 2*np.pi)
            cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])

        
        

        
        
        





def draw_mp4(model: ShortRangePhaseInter1, savePath: str = "./data", mp4Path: str = "./mp4", step: int = 1, earlyStop: int = None):

    targetPath = f"{savePath}/{model}.h5"
    totalPositionX = pd.read_hdf(targetPath, key="positionX")
    totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
    TNum = totalPositionX.shape[0] // model.agentsNum
    totalPositionX = totalPositionX.values.reshape(TNum, model.agentsNum, 2)
    totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, model.agentsNum)
    sa = StateAnalysis(showTqdm=True)

    if earlyStop is not None:
        totalPositionX = totalPositionX[:earlyStop]
        totalPhaseTheta = totalPhaseTheta[:earlyStop]
        TNum = earlyStop
    maxAbsPos = np.max(np.abs(totalPositionX))

    def plot_frame(i):
        positionX = totalPositionX[i]
        phaseTheta = totalPhaseTheta[i]

        fig.clear()
        fig.subplots_adjust(left=0.15, right=1, bottom=0.1, top=0.95)
        ax1 = plt.subplot(1, 1, 1)
        model.positionX = positionX
        model.phaseTheta = phaseTheta
        model.counts = i * model.shotsnaps
        StateAnalysis.plot_last_state(sa, model, ax1)
        ax1.set_xlim(-maxAbsPos, maxAbsPos)
        ax1.set_ylim(-maxAbsPos, maxAbsPos)
        roundBound = np.round(maxAbsPos)
        ax1.set_xticks([-roundBound, -roundBound / 2, 0, roundBound / 2, roundBound])
        ax1.set_yticks([-roundBound, -roundBound / 2, 0, roundBound / 2, roundBound])
        plt.tight_layout()
        pbar.update(1)

    frames = np.arange(0, TNum, step)
    pbar = tqdm(total=len(frames) + 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ani = ma.FuncAnimation(fig, plot_frame, frames=frames, interval=100, repeat=False)
    ani.save(f"{mp4Path}/{model}.mp4", dpi=150)
    plt.close()

    pbar.close()