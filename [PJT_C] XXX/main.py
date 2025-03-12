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
import warnings
warnings.filterwarnings("ignore")

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

# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'
if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"

sys.path.append("..")
from swarmalatorlib.template import Swarmalators2D


class MobileDrive(Swarmalators2D):
    def __init__(self, agentsNum: int, K: float, J: float, F: float,
                 randomSeed: int = 100, dt: float = 0.1, tqdm: bool = True, 
                 savePath: str = None, shotsnaps: int = 10, overWrite: bool = False) -> None:
        super().__init__(agentsNum=agentsNum, dt=dt, K=K, randomSeed=randomSeed, tqdm=tqdm, 
                         savePath=savePath, shotsnaps=shotsnaps, overWrite=overWrite)
        self.J = J
        self.F = F
        self.driveThateVelocityOmega = 0.5 * np.pi
        self.driveAngularVelocityW = 0.5
        self.druveRadiusR = 0.5
        self.drivePosition = np.array([0.5, 0])
        self.drivePhase = 0
        # self.drivePhasePool = []
        self.one = np.ones((agentsNum, agentsNum))
        self.temp["pointX"] = np.zeros((agentsNum, 2)) * np.nan
        self.temp["pointTheta"] = np.zeros(agentsNum) * np.nan

    @property
    def Fatt(self) -> np.ndarray:
        """
        Effect of phase similarity on spatial attraction: 1 + J * cos(theta_j - theta_i)
        """
        return 1 + self.J * np.cos(self.temp["deltaTheta"])

    @property
    def Frep(self) -> np.ndarray:
        """Effect of phase similarity on spatial repulsion: 1"""
        return self.one
    
    @property
    def Iatt(self) -> np.ndarray:
        """Spatial attraction: (x_j - x_i) / |x_j - x_i|"""
        return self.div_distance_power(numerator=self.temp["deltaX"], power=1)
    
    @property
    def Irep(self) -> np.ndarray:
        """Spatial repulsion: (x_j - x_i) / |x_j - x_i| ^ 2"""
        return self.div_distance_power(numerator=self.temp["deltaX"], power=2)
    
    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity: 0"""
        return 0
    
    @property
    def omega(self) -> np.ndarray:
        """Natural frequency: 0"""
        return 0
    
    @property
    def H(self) -> np.ndarray:
        """Phase interaction: sin(theta_j - theta_i)"""
        return np.sin(self.temp["deltaTheta"])
    
    @property
    def G(self) -> np.ndarray:
        """Effect of spatial similarity on phase couplings: 1 / |x_i - x_j|"""
        return self.div_distance_power(numerator=self.one, power=1, dim=1)
    
    @property
    def P(self) -> np.ndarray:
        """External drive: F*cos(omega*t - theta_i) / |x_0 - x_i|"""
        disance = self.distance_x(((self.positionX - self.drivePosition)[:, np.newaxis]))[:, 0]
        
        return self.F * np.cos(self.drivePhase - self.phaseTheta) / disance

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="pointX", value=pd.DataFrame(self.temp["pointX"]))
            self.store.append(key="pointTheta", value=pd.DataFrame(self.temp["pointTheta"]))
            self.store.append(key="drivePosAndPhs", value=pd.DataFrame(np.concatenate([self.drivePosition, [self.drivePhase]])))

    def update(self) -> None:
        self.update_temp()
        pointX, pointTheta = self._calc_point(
            self.positionX, self.phaseTheta,
            self.velocity, self.omega,
            self.Iatt, self.Irep,
            self.Fatt, self.Frep,
            self.H, self.G, self.P,
            self.K, self.dt, 
        )
        self.temp["pointX"] = pointX
        self.temp["pointTheta"] = pointTheta
        self.positionX += pointX * self.dt
        self.phaseTheta = np.mod(self.phaseTheta + pointTheta * self.dt, 2 * np.pi)
        t = self.counts * self.dt
        self.drivePosition = np.array([
            self.druveRadiusR * np.cos(self.driveAngularVelocityW * t),
            self.druveRadiusR * np.sin(self.driveAngularVelocityW * t)
        ])
        self.drivePhase = self.driveThateVelocityOmega * t
        
        self.counts += 1

    @staticmethod
    @nb.njit
    def _calc_point(
        positionX: np.ndarray, phaseTheta: np.ndarray,
        velocity: np.ndarray, omega: np.ndarray,
        Iatt: np.ndarray, Irep: np.ndarray,
        Fatt: np.ndarray, Frep: np.ndarray,
        H: np.ndarray, G: np.ndarray, P: np.ndarray,
        K: float, dt: float
    ):
        dim = positionX.shape[0]
        pointX = velocity + np.sum(
            Iatt * Fatt.reshape((dim, dim, 1)) - Irep * Frep.reshape((dim, dim, 1)),
            axis=1
        ) / dim
        
        pointTheta = omega + K * np.sum(H * G, axis=1) / dim + P
        
        return pointX, pointTheta

    def plot(self, ax: plt.Axes = None) -> None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(self.drivePosition[0], self.drivePosition[1], c='r', s=100, marker='x', zorder=10)
        sc = ax.scatter(self.positionX[:, 0], self.positionX[:, 1],
                    c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)

        cbar = plt.colorbar(sc, ticks=[0, np.pi, 2*np.pi], ax=ax)
        cbar.ax.set_ylim(0, 2*np.pi)
        cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])

    def __str__(self) -> str:
        return f"MobileDrive_a{self.agentsNum}_K{self.K:.2f}_J{self.J:.2f}_F{self.F:.2f}"
    

class StateAnalysis:
    def __init__(self, model: MobileDrive = None, lookIndex: int = -1, showTqdm: bool = False):
        self.lookIndex = lookIndex
        self.showTqdm = showTqdm

        if model is not None:
            self.model = model
            targetPath = f"{self.model.savePath}/{self.model}.h5"
            totalPositionX = pd.read_hdf(targetPath, key="positionX")
            totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
            totalPointX = pd.read_hdf(targetPath, key="pointX")
            totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
            totalDrivePosAndPhs = pd.read_hdf(targetPath, key="drivePosAndPhs")
            
            TNum = totalPositionX.shape[0] // self.model.agentsNum
            self.TNum = TNum
            self.tRange = np.arange(0, (TNum - 1) * model.shotsnaps, model.shotsnaps) * self.model.dt
            self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
            self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)
            self.totalPointX = totalPointX.values.reshape(TNum, self.model.agentsNum, 2)
            self.totalPointTheta = totalPointTheta.values.reshape(TNum, self.model.agentsNum)
            totalDrivePosAndPhs = totalDrivePosAndPhs.values.reshape(TNum, 3)
            self.totalDrivePosition = totalDrivePosAndPhs[:, :2]
            self.totalDrivePhaseTheta = totalDrivePosAndPhs[:, 2]

            if self.showTqdm:
                self.iterObject = tqdm(range(1, self.totalPhaseTheta.shape[0]))
            else:
                self.iterObject = range(1, self.totalPhaseTheta.shape[0])

    def get_state(self, index: int = -1):
        return self.totalPositionX[index], self.totalPhaseTheta[index], self.totalDrivePosition[index], self.totalDrivePhaseTheta[index]

    @staticmethod
    def calc_order_parameter_R(model: MobileDrive) -> float:
        return np.abs(np.sum(np.exp(1j * model.phaseTheta))) / model.agentsNum
    
    @staticmethod
    def calc_order_parameter_S(model: MobileDrive) -> float:
        phi = np.arctan2(model.positionX[:, 1], model.positionX[:, 0])
        Sadd = np.abs(np.sum(np.exp(1j * (phi + model.phaseTheta)))) / model.agentsNum
        Ssub = np.abs(np.sum(np.exp(1j * (phi - model.phaseTheta)))) / model.agentsNum
        return np.max([Sadd, Ssub])

    @staticmethod
    def calc_order_parameter_Vp(model: MobileDrive) -> float:
        pointX = model.temp["pointX"]
        phi = np.arctan2(pointX[:, 1], pointX[:, 0])
        return np.abs(np.sum(np.exp(1j * phi))) / model.agentsNum
    
    @staticmethod
    def calc_order_parameter_Ptr(model: MobileDrive) -> float:
        pointTheta = model.temp["pointTheta"]
        Ntr = np.abs(pointTheta - model.driveThateVelocityOmega) < 0.2 / model.dt * 0.1
        return Ntr.sum() / model.agentsNum
    
    def plot_last_state(self, model: MobileDrive = None, ax: plt.Axes = None, withColorBar: bool =True, withDriver: bool = True,
                        s: float = 50, driveS: float = 100) -> None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        if model is not None:
            t = model.counts * model.dt
            drivePosition = np.array([
                np.cos(model.driveThateVelocityOmega * t) * model.druveRadiusR,
                np.sin(model.driveThateVelocityOmega * t) * model.druveRadiusR
            ])
            if withDriver:
                ax.scatter(drivePosition[0], drivePosition[1], color="white", s=driveS, marker='o', edgecolors='k', zorder=10)
                driveCircle = plt.Circle((0, 0), model.druveRadiusR, color='black', fill=False, lw=2, linestyle='--')
                ax.add_artist(driveCircle)
            sc = ax.scatter(model.positionX[:, 0], model.positionX[:, 1], s=s,
                            c=model.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
            maxPos = np.abs(model.positionX).max()
        else:
            if withDriver:
                ax.scatter(self.totalDrivePosition[self.lookIndex, 0], self.totalDrivePosition[self.lookIndex, 1], 
                           color="white", s=driveS, marker='o', edgecolors='k', zorder=10)
                driveCircle = plt.Circle((0, 0), self.model.druveRadiusR, color='black', fill=False, lw=2, linestyle='--')
                ax.add_artist(driveCircle)
            sc = ax.scatter(self.totalPositionX[self.lookIndex, :, 0], self.totalPositionX[self.lookIndex, :, 1], s=s,
                            c=self.totalPhaseTheta[self.lookIndex], cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)
            maxPos = np.abs(self.totalPositionX[self.lookIndex]).max()
            # print(maxPos)
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