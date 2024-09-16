import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm.notebook import tqdm
from itertools import product
import numpy as np
import numba as nb
import imageio
import os
import shutil

randomSeed = 100


new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

import seaborn as sns

sns.set(font_scale=1.1, rc={
    'figure.figsize': (10, 6),
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


class Swarmalators3D():
    def __init__(self, agentsNum: int, dt: float, K: float) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 3)) * 2 - 1
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.K = K

    @staticmethod
    @nb.njit
    def _delta_theta(phaseTheta):
        dim = phaseTheta.shape[0]
        subTheta = phaseTheta - np.repeat(phaseTheta, dim).reshape(dim, dim)

        deltaTheta = np.zeros((dim, dim - 1))
        for i in np.arange(dim):
            deltaTheta[i, :i], deltaTheta[i, i:] = subTheta[i, :i], subTheta[i, i + 1 :]
        return deltaTheta

    @property
    def deltaTheta(self) -> np.ndarray:
        """Phase difference between agents"""
        return self._delta_theta(self.phaseTheta)
    
    @staticmethod
    @nb.njit
    def _delta_x(positionX):
        dim = positionX.shape[0]
        subX = positionX - np.repeat(positionX, dim).reshape(dim, 3, dim).transpose(0, 2, 1)
        deltaX = np.zeros((dim, dim - 1, 3))
        for i in np.arange(dim):
            deltaX[i, :i], deltaX[i, i:] = subX[i, :i], subX[i, i + 1 :]
        return deltaX

    @property
    def deltaX(self) -> np.ndarray:
        """Spatial difference between agents"""
        return self._delta_x(self.positionX)

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
    def distance_x(deltaX):
        return np.sqrt(
            deltaX[:, :, 0] ** 2 + deltaX[:, :, 1] ** 2 + deltaX[:, :, 2] ** 2
        )
    
    @staticmethod
    @nb.njit
    def distance_x_2(deltaX):
        return np.sqrt(
            deltaX[:, :, 0] ** 2 + deltaX[:, :, 1] ** 2 + deltaX[:, :, 2] ** 2
        ).reshape(deltaX.shape[0], deltaX.shape[1], 1)

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
            Iatt * Fatt.reshape((dim, dim - 1, 1)) - Irep * Frep.reshape((dim, dim - 1, 1)),
            axis=1
        ) / (dim - 1)
        positionX += pointX * dt
        pointTheta = omega + K * np.sum(H * G, axis=1) / (dim - 1)
        phaseTheta = np.mod(phaseTheta + pointTheta * dt, 2 * np.pi)
        return positionX, phaseTheta
    

    def update(self) -> None:
        self.positionX, self.phaseTheta = self._update(
            self.positionX, self.phaseTheta,
            self.velocity, self.omega,
            self.Iatt, self.Irep,
            self.Fatt, self.Frep,
            self.H, self.G,
            self.K, self.dt
        )

class ChiralSwarmalators3D(Swarmalators3D):
    def __init__(self, agentsNum: int, dt: float,
                 J: float, K: float,
                 A: float, B: float) -> None:
        super().__init__(agentsNum, dt, K)
        self.J = J
        self.K = K
        self.A = A
        self.B = B * np.ones((agentsNum, agentsNum - 1))
        self.omegaValue = np.ones(agentsNum)
        randomHalfIdx = np.random.choice(agentsNum, agentsNum // 2, replace=False)
        self.omegaValue[randomHalfIdx] = -1

    @property
    def omega(self) -> np.ndarray:
        return self.omegaValue

    @property
    def Fatt(self) -> np.ndarray:
        """
        Effect of phase similarity on spatial attraction
        A + J * cos(theta_j - theta_i)
        """
        return self.A + self.J * np.cos(self.deltaTheta)
    
    @property
    def Frep(self) -> np.ndarray:
        """Effect of phase similarity on spatial repulsion: B"""
        return self.B

    @property
    def Iatt(self) -> np.ndarray:
        """Spatial attraction: (x_j - x_i) / |x_j - x_i|"""

        return self.deltaX / self.distance_x_2(self.deltaX)
    # self.deltaX / np.linalg.norm(self.deltaX, axis=-1, keepdims=True)
    
    @property
    def Irep(self) -> np.ndarray:
        """Spatial repulsion: (x_j - x_i) / |x_j - x_i| ^ 2"""
        return self.deltaX / self.distance_x_2(self.deltaX) ** 2

    @property
    def H(self) -> np.ndarray:
        """Phase interaction: sin(theta_j - theta_i)"""
        return np.sin(self.deltaTheta)
    
    @property
    def G(self) -> np.ndarray:
        """Effect of spatial similarity on phase couplings: 1 / |x_i - x_j|"""
        return 1 / self.distance_x(self.deltaX)
    
    @staticmethod
    @nb.njit
    def _velocity(omega: np.ndarray, phaseTheta: np.ndarray) -> np.ndarray:
        dim = omega.shape[0]
        n = np.zeros((dim, 3))
        omegaSign = np.sign(omega)
        n[:, 0] = np.cos(phaseTheta + np.pi / 2) * omegaSign
        n[:, 1] = np.sin(phaseTheta + np.pi / 2) * omegaSign
        n[:, 2] = np.cos(phaseTheta) * (omegaSign == 1) + np.sin(phaseTheta) * (omegaSign == -1)
        return n

    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity: 0"""
        return self._velocity(self.omega, self.phaseTheta)
    

class ChiralSwarmalators3DUrand(ChiralSwarmalators3D):
    def __init__(self, agentsNum: int, dt: float,
                 J: float, K: float,
                 A: float, B: float) -> None:
        super().__init__(agentsNum, dt, J, K, A, B)
        self.zi = np.random.random(agentsNum)

    @staticmethod
    @nb.njit
    def _velocity(omega: np.ndarray, phaseTheta: np.ndarray, zi: np.ndarray) -> np.ndarray:
        dim = omega.shape[0]
        n = np.zeros((dim, 3))
        omegaSign = np.sign(omega)
        n[:, 0] = np.cos(phaseTheta + np.pi / 2) * omegaSign
        n[:, 1] = np.sin(phaseTheta + np.pi / 2) * omegaSign
        n[:, 2] = zi
        return n

    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity: 0"""
        return self._velocity(self.omega, self.phaseTheta, self.zi)



class FreqCoupledChiralSwarmalators3D(ChiralSwarmalators3D):
    def __init__(self, agentsNum: int, dt: float,
                 J: float, K: float,
                 A: float, B: float) -> None:
        super().__init__(agentsNum, dt, J, K, A, B)
        self.QdotX = np.pi / 2 * self.delta_sign_omega(self.omegaValue)
        self.QdotTheta = np.pi / 4 * self.delta_sign_omega(self.omegaValue)

    @staticmethod
    @nb.njit
    def delta_sign_omega(omega: np.ndarray):
        dim = omega.shape[0]
        subOmega = np.sign(omega) - np.sign(np.repeat(omega, dim).reshape(dim, dim))

        deltaOmega = np.zeros((dim, dim - 1))
        for i in np.arange(dim):
            deltaOmega[i, :i], deltaOmega[i, i:] = subOmega[i, :i], subOmega[i, i + 1 :]
        return np.abs(deltaOmega)
    
    @property
    def Fatt(self) -> np.ndarray:
        """
        Effect of phase similarity on spatial attraction
        A + J * cos(theta_j - theta_i)
        """
        return self.A + self.J * np.cos(self.deltaTheta - self.QdotX)
    
    @property
    def H(self) -> np.ndarray:
        """Phase interaction: sin(theta_j - theta_i)"""
        return np.sin(self.deltaTheta - self.QdotTheta)


class FreqCoupledChiralSwarmalators3DUrand(FreqCoupledChiralSwarmalators3D):
    def __init__(self, agentsNum: int, dt: float,
                 J: float, K: float,
                 A: float, B: float) -> None:
        super().__init__(agentsNum, dt, J, K, A, B)
        self.zi = np.random.random(agentsNum)

    @staticmethod
    @nb.njit
    def _velocity(omega: np.ndarray, phaseTheta: np.ndarray, zi: np.ndarray) -> np.ndarray:
        dim = omega.shape[0]
        n = np.zeros((dim, 3))
        omegaSign = np.sign(omega)
        n[:, 0] = np.cos(phaseTheta + np.pi / 2) * omegaSign
        n[:, 1] = np.sin(phaseTheta + np.pi / 2) * omegaSign
        n[:, 2] = zi
        return n

    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity: 0"""
        return self._velocity(self.omega, self.phaseTheta, self.zi)


def plot_3d_model(positionX, phaseTheta, velocity, xlim, ylim, zlim, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 画3D向量，以positionX为起点，以velocity为方向，颜色为相位，保持箭头和剑身颜色一致
    colors = [[new_cmap(i)] * 3 for i in colors_idx(phaseTheta)]
    for i in range(len(colors)):
        ax.quiver(positionX[i, 0], positionX[i, 1], positionX[i, 2],
                  velocity[i, 0], velocity[i, 1], velocity[i, 2],
                  colors=colors[i],
                  alpha=0.8, length=0.25, arrow_length_ratio=0.5, linewidths=2)
    scatter = ax.scatter(positionX[:, 0], positionX[:, 1], positionX[:, 2],
                         c=phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2*np.pi, s=0.01)
    cbar = plt.colorbar(scatter, ticks=[0, np.pi, 2*np.pi])
    cbar.ax.set_ylim(0, 2*np.pi)
    cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_title(title)


def run_J_K_model(model, T: int = 1000):

    J = model.J
    K = model.K

    imagePath = f"images/images_{J}_{K}"
    if os.path.exists(imagePath):
        shutil.rmtree(imagePath)
    os.mkdir(imagePath)

    positions = []
    phases = []
    velocities = []

    for i in tqdm(range(T), desc=f"J={J}, K={K}"):
        model.update()
        if i % 5 == 0:
            positions.append(model.positionX.copy())
            phases.append(model.phaseTheta.copy())
            velocities.append(model.velocity.copy())    

    xOfPositions = np.array(positions)[:, :, 0]
    yOfPositions = np.array(positions)[:, :, 1]
    zOfPositions = np.array(positions)[:, :, 2]
    xlim = (np.min(xOfPositions), np.max(xOfPositions))
    ylim = (np.min(yOfPositions), np.max(yOfPositions))
    zlim = (np.min(zOfPositions), np.max(zOfPositions))
    images = []
    del xOfPositions, yOfPositions, zOfPositions

    for i, (position, phase, velocity) in tqdm(enumerate(zip(positions, phases, velocities)), total=len(positions)):
        plot_3d_model(position, phase, velocity, xlim=xlim, ylim=ylim, zlim=zlim, title=f"{i}. J={J}, K={K}, t={0.5 * i}")
        plt.savefig(f"{imagePath}/{i}.png")
        plt.close()
        images.append(imageio.imread(f"{imagePath}/{i}.png"))
        
    # 创建以model.__name__命名的文件夹，并创建gif和npz子文件夹
    modelName = model.__class__.__name__
    if not os.path.exists(f"./{modelName}"):
        os.mkdir(f"./{modelName}")
    if not os.path.exists(f"./{modelName}/gif"):
        os.mkdir(f"./{modelName}/gif")
    if not os.path.exists(f"./{modelName}/data"):
        os.mkdir(f"./{modelName}/data")

    imageio.mimsave(f"./{modelName}/gif/J={J}_K={K}.gif", images, fps=10)
    del images
    np.savez(f"./{modelName}/data/J={J}_K={K}.npz", positions=positions, phases=phases, velocities=velocities)


def run_multi_process(Js, Ks, processNum: int, modelClass: object):
    
    models = []

    for J, K in product(Js, Ks):
        
        models.append(
            modelClass(agentsNum=500, dt=0.1, J=J, K=K, A=1, B=1)
        )

    with Pool(processNum) as pool:
        _ = list(tqdm(
            pool.imap(run_J_K_model, models),
            total=len(models)
        )) 
        pool.close()

