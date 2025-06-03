import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import numba as nb
import numpy as np
import warnings
import sys
import os

if "ipykernel_launcher.py" in sys.argv[0]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
)
if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/Programs/ffmpeg/bin/ffmpeg.exe"


class Swarmalators:
    def __init__(self, agentsNum: int, dt: float, K: float, randomSeed: int = 100,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, overWrite: bool = False) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * 2 - 1
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.K = K
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.temp = {}
        self.overWrite = overWrite 

    def init_store(self):
        if self.savePath is None:
            self.store = None
        else:
            if os.path.exists(f"{self.savePath}/{self}.h5"):
                if self.overWrite:
                    os.remove(f"{self.savePath}/{self}.h5")
                else:
                    print(f"{self.savePath}/{self}.h5 already exists")
                    return False
            self.store = pd.HDFStore(f"{self.savePath}/{self}.h5")
        self.append()
        return True


    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))

    @property
    def deltaTheta(self) -> np.ndarray:
        """Phase difference between agents"""
        return self.phaseTheta - self.phaseTheta[:, np.newaxis]

    @property
    def deltaX(self) -> np.ndarray:
        """
        Spatial difference between agents

        Shape: (agentsNum, agentsNum, 2)

        Every cell = otherAgent - agentSelf !!!
        """
        return self.positionX - self.positionX[:, np.newaxis]

    @property
    def Fatt(self) -> np.ndarray:
        """Effect of phase similarity on spatial attraction"""
        pass

    @property
    def Frep(self) -> np.ndarray:
        """Effect of phase similarity on spatial repulsion"""
        pass

    @property
    def Iatt(self) -> np.ndarray:
        """Spatial attraction"""
        pass

    @property
    def Irep(self) -> np.ndarray:
        """Spatial repulsion"""
        pass

    @property
    def H(self) -> np.ndarray:
        """Phase interaction"""
        pass

    @property
    def G(self) -> np.ndarray:
        """Effect of spatial similarity on phase couplings"""
        pass

    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity"""
        pass

    @property
    def omega(self) -> np.ndarray:
        """Natural intrinsic frequency"""
        pass

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
        pass

    def update_temp(self):
        pass

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

    def run(self, TNum: int):
        
        if not self.init_store():
            return 

        if self.tqdm:
            iterRange = tqdm(range(TNum))
        else:
            iterRange = range(TNum)

        for idx in iterRange:
            self.update()
            self.append()
            self.counts = idx

        self.close()

    def plot(self) -> None:
        pass

    def close(self):
        if self.store is not None:
            self.store.close()

    def __repr__(self):
        return str(self)


class Swarmalators1D(Swarmalators):
    def __init__(self, agentsNum: int, dt: float, K: float, randomSeed: int = 100,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, overWrite: bool = False) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random(agentsNum) * np.pi / 2 # - np.pi
        self.phaseTheta = np.random.random(agentsNum) * np.pi / 2  # - np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.K = K
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.temp = {}
        self.overWrite = overWrite 

    def plot(self) -> None:
        plt.figure(figsize=(6, 5))

        plt.scatter(
            np.cos(self.positionX), np.sin(self.positionX),
            c=self.phaseTheta, cmap=new_cmap, alpha=0.8, 
            vmin=-np.pi, vmax=np.pi
        )
        circle = plt.Circle((0, 0), 1, color="black", fill=False)
        plt.gca().add_patch(circle)

        cbar = plt.colorbar(ticks=[-np.pi, 0, np.pi])
        cbar.ax.set_ylim(-np.pi, np.pi)
        cbar.ax.set_yticklabels(['$-\pi$', '$0$', '$\pi$'])


    def update_temp(self):
        self.temp["deltaTheta"] = self.deltaTheta
        self.temp["deltaX"] = self.deltaX

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
        pointX = velocity + np.sum(Iatt * Fatt, axis=1) / positionX.shape[0]
        positionX += pointX * dt
        positionX = np.mod(positionX, np.pi / 0.5)
        # positionX = np.mod(positionX + np.pi, 2 * np.pi) - np.pi
        pointTheta = omega + K * np.sum(H * G, axis=1) / positionX.shape[0]
        phaseTheta += pointTheta * dt
        phaseTheta = np.mod(phaseTheta, np.pi / 0.5)
        # phaseTheta = np.mod(phaseTheta + np.pi, 2 * np.pi) - np.pi
        return positionX, phaseTheta
        

class Swarmalators2D(Swarmalators):
    def __init__(self, agentsNum: int, dt: float, K: float, randomSeed: int = 100,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, overWrite: bool = False) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * 2 - 1
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.K = K
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.temp = {}
        self.overWrite = overWrite 
        self.update_temp()

    def plot(self, ax: plt.Axes = None) -> None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        sc = ax.scatter(self.positionX[:, 0], self.positionX[:, 1],
                    c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi)

        cbar = plt.colorbar(sc, ticks=[0, np.pi, 2*np.pi], ax=ax)
        cbar.ax.set_ylim(0, 2*np.pi)
        cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])

    def update_temp(self):
        self.temp["deltaTheta"] = self.deltaTheta
        self.temp["deltaX"] = self.deltaX
        self.temp["distanceX"] = self.distance_x(self.temp["deltaX"])
        self.temp["distanceX2"] = self.temp["distanceX"].reshape(self.agentsNum, self.agentsNum, 1)

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
        positionX += pointX * dt
        pointTheta = omega + K * np.sum(H * G, axis=1) / dim
        phaseTheta = np.mod(phaseTheta + pointTheta * dt, 2 * np.pi)
        return positionX, phaseTheta
    
    @staticmethod
    @nb.njit
    def distance_x_2(deltaX):
        return np.sqrt(deltaX[:, :, 0] ** 2 + deltaX[:, :, 1] ** 2).reshape(deltaX.shape[0], deltaX.shape[1], 1)

    @staticmethod
    @nb.njit
    def distance_x(deltaX):
        return np.sqrt(deltaX[:, :, 0] ** 2 + deltaX[:, :, 1] ** 2)
    
    # @staticmethod
    # @nb.njit
    # def _delta_theta(phaseTheta):
    #     dim = phaseTheta.shape[0]
    #     subTheta = phaseTheta - np.repeat(phaseTheta, dim).reshape(dim, dim)

    #     deltaTheta = np.zeros((dim, dim - 1))
    #     for i in np.arange(dim):
    #         deltaTheta[i, :i], deltaTheta[i, i:] = subTheta[i, :i], subTheta[i, i + 1 :]
    #     return deltaTheta

    # @staticmethod
    # @nb.njit
    # def _delta_x(positionX):
    #     dim = positionX.shape[0]
    #     subX = positionX - np.repeat(positionX, dim).reshape(dim, 2, dim).transpose(0, 2, 1)
    #     deltaX = np.zeros((dim, dim - 1, 2))
    #     for i in np.arange(dim):
    #         deltaX[i, :i], deltaX[i, i:] = subX[i, :i], subX[i, i + 1 :]
    #     return deltaX
    
    @staticmethod
    @nb.njit
    def _calc_div_distance_power(numerator: np.ndarray, denominator: np.ndarray, powerOfDeno: float):
        return numerator / (denominator ** powerOfDeno)

    def div_distance_power(self, numerator: np.ndarray, power: float, dim: int = 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if dim == 2:
                answer = self._calc_div_distance_power(numerator, self.temp["distanceX2"], power)
            else:  # dim == 1
                answer = self._calc_div_distance_power(numerator, self.temp["distanceX"], power)
            
        answer[np.isnan(answer) | np.isinf(answer)] = 0

        return answer