import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
import numba as nb
import numpy as np
import os
import matplotlib.animation as ma



new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)
if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams['animation.ffmpeg_path'] = "D:/ProgramData/ffmpeg/bin/ffmpeg.exe"


class AdaptiveInteraction2D:
    def __init__(self, agentsNum: int, dt: float, 
                 J: float, K:float,
                 r_c: float,
                 randomSeed: int = 100,
                 distribution: str = None,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * 2 - 1
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        self.agentsNum = agentsNum
        self.dt = dt

        self.J = J
        self.K = K

        self.A_ij = np.random.random((agentsNum, agentsNum)) * 2 - 1
        self.r_c = r_c

        self.distribution = distribution
        if self.distribution is None:
            self.omegaValue = 0
        elif self.distribution == "uniform0.1":
            self.omegaValue = np.random.uniform(-0.1, 0.1, self.agentsNum)
        elif self.distribution == "uniform1":
            self.omegaValue = np.random.uniform(-1, 1, self.agentsNum)
        elif self.distribution == "normal":
            self.omegaValue = np.random.normal(0, 1, self.agentsNum)

        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.temp = {}

        self.one = np.ones((agentsNum, agentsNum))
        self.temp["pointX"] = np.zeros((agentsNum, 2)) 
        self.temp["pointTheta"] = np.zeros(agentsNum) * np.nan

        self.filename = f"Adaptive2D_J{self.J:.2f}_K{self.K:.2f}_rc{self.r_c:.2f}_Num{self.agentsNum}"

    def __str__(self) -> str:
        return self.filename
    
    def init_store(self):
        if self.savePath is None:
            self.store = None
        else:
            filename = f"{self.filename}.h5"
            if os.path.exists(f"{self.savePath}/{filename}"):
                os.remove(f"{self.savePath}/{filename}")
            self.store = pd.HDFStore(f"{self.savePath}/{filename}")
        self.append()

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta",value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="pointX", value=pd.DataFrame(self.temp["pointX"]))
            self.store.append(key="pointTheta", value=pd.DataFrame(self.temp["pointTheta"]))
            self.store.append(key="A_ij", value=pd.DataFrame(self.A_ij))

    @property
    def deltaTheta(self) -> np.ndarray:
        return self.phaseTheta - self.phaseTheta[:, np.newaxis]

    @property
    def deltaX(self) -> np.ndarray:
        return self.positionX - self.positionX[:, np.newaxis]
    
    @property
    def phi(self) -> np.ndarray:
        return np.arctan2(self.positionX[:, 1], self.positionX[:, 0])

    @property
    def deltaPhi(self) -> np.ndarray:
        return self.phi - self.phi[:, np.newaxis]

    @staticmethod
    @nb.njit
    def distance_x(deltaX):
        return np.sqrt(deltaX[:, :, 0] ** 2 + deltaX[:, :, 1] ** 2)

    def div_distance_power(self, numerator: np.ndarray, power: float, dim: int = 2):
        if dim == 2:
            answer = numerator / self.temp["distanceX2"] ** power
        else:
            answer = numerator / self.temp["distanceX"] ** power

        answer[np.isnan(answer) | np.isinf(answer)] = 0

        return answer
    
    def update_temp(self):
        self.temp["deltaTheta"] = self.deltaTheta
        self.temp["deltaX"] = self.deltaX
        self.temp["deltaPhi"] = self.deltaPhi
        self.temp["distanceX"] = self.distance_x(self.temp["deltaX"])
        self.temp["distanceX2"] = self.temp["distanceX"].reshape(self.agentsNum, self.agentsNum, 1)

    @property
    def Fatt(self) -> np.ndarray:
        return 1 + self.J * np.cos(self.temp["deltaTheta"])

    @property
    def Frep(self) -> np.ndarray:
        return self.one
    
    @property
    def Iatt(self) -> np.ndarray:
        return self.div_distance_power(numerator=self.temp["deltaX"], power=1)
    
    @property
    def Irep(self) -> np.ndarray:
        return self.div_distance_power(numerator=self.temp["deltaX"], power=2)
    
    @property
    def velocity(self) -> np.ndarray:
        return 0
    
    @property
    def omega(self) -> np.ndarray:
        return self.omegaValue
    
    @property
    def H(self) -> np.ndarray:
        return np.sin(self.temp["deltaTheta"])
        # return np.sin(self.temp["deltaTheta"] - self.temp["deltaPhi"])
    
    @property
    def G(self) -> np.ndarray:
        return self.div_distance_power(numerator=self.one, power=1, dim=1)
        # return self.one
    
    @property
    def A_ij_dot(self):
        return (self.r_c-self.r_ij) * (self.one + self.A_ij) * (self.one-self.A_ij)
    
    @property
    def r_ij(self):
        return np.sqrt((1+np.cos(self.temp["deltaPhi"]))/2)
    
    @staticmethod
    @nb.njit
    def _calc_point(
        positionX: np.ndarray, phaseTheta: np.ndarray,
        velocity: np.ndarray, omega: np.ndarray,
        Iatt: np.ndarray, Irep: np.ndarray,
        Fatt: np.ndarray, Frep: np.ndarray,
        H: np.ndarray, G: np.ndarray,
        dt: float,K: float,
        A_ij: np.ndarray, A_ij_dot: np.ndarray,
        r_ij: np.ndarray, r_c: float   
    ):
        dim = positionX.shape[0]
        pointX = velocity + np.sum(
            Iatt * Fatt.reshape((dim, dim, 1)) - Irep * Frep.reshape((dim, dim, 1)),
            axis=1
        ) / dim 
        pointTheta = omega + K * np.sum(A_ij * H * G, axis=1) / dim
        A_ij += A_ij_dot*dt

        return pointX, pointTheta
    
    def update(self) -> None:
        self.update_temp()
        self.pointX, self.pointTheta = self._calc_point(
            self.positionX, self.phaseTheta,
            self.velocity, self.omega,
            self.Iatt, self.Irep,
            self.Fatt, self.Frep,
            self.H, self.G,
            self.dt,self.K,
            self.A_ij, self.A_ij_dot,
            self.r_ij, self.r_c
        )
        self.temp["pointX"] = self.pointX
        self.temp["pointTheta"] = self.pointTheta
        self.positionX += self.pointX * self.dt
        self.phaseTheta = np.mod(self.phaseTheta + self.pointTheta * self.dt, 2 * np.pi)

        self.counts += 1

    def run(self, TNum: int):
        self.init_store()
        if self.tqdm:
            global pbar
            pbar = tqdm(total=TNum)
        for i in np.arange(TNum):
            self.update()
            self.append()
            if self.tqdm:
                pbar.update(1)
        if self.tqdm:
            pbar.close()
        self.close()

    def close(self):
        if self.store is not None:
            self.store.close()
    # 作图
    def plot(self, ax: plt.Axes = None) -> None:
        fig,ax = plt.subplots(figsize=(6, 5))
        maxAbsPos = np.max(np.abs(self.positionX))
        # N = np.sqrt(self.pointX[:,0]**2 + self.pointX[:,1]**2)
        # U,V = self.pointX[:,0]/N, self.pointX[:,1]/N
        # qv = plt.quiver(self.positionX[:, 0], self.positionX[:, 1], U, 
        #                 V,self.phaseTheta,cmap='viridis', clim=(0, 2*np.pi))
        scatter = plt.scatter(self.positionX[:, 0], self.positionX[:, 1], 
                              c=self.phaseTheta, cmap='viridis', clim=(0, 2*np.pi))
        ax.set_xlim(-maxAbsPos, maxAbsPos)
        ax.set_ylim(-maxAbsPos, maxAbsPos)
        ax.set_title(f"Time: {self.counts*self.dt:.2f}")

        cbar = plt.colorbar(scatter, ticks=[0, np.pi, 2*np.pi], ax=ax)
        cbar.ax.set_ylim(0, 2*np.pi)
        cbar.ax.set_yticklabels(['$0$', '$\pi$', '$2\pi$'])

class StateAnalysis:
    def __init__(self, model: AdaptiveInteraction2D, lookIndex: int = -1, showTqdm: bool = False):
        self.model = model
        self.lookIndex = lookIndex
        self.showTqdm = showTqdm
        
        targetPath = f"{self.model.savePath}/{self.model}.h5"
        totalPositionX = pd.read_hdf(targetPath, key="positionX")
        totalPhaseTheta = pd.read_hdf(targetPath, key="phaseTheta")
        totalPointX = pd.read_hdf(targetPath, key="pointX")
        totalPointTheta = pd.read_hdf(targetPath, key="pointTheta")
        TNum = totalPositionX.shape[0] // self.model.agentsNum

        self.TNum = TNum
        self.tRange = np.arange(0, (TNum - 1) * model.shotsnaps, model.shotsnaps) * self.model.dt

        self.totalPositionX = totalPositionX.values.reshape(TNum, self.model.agentsNum, 2)
        self.totalPhaseTheta = totalPhaseTheta.values.reshape(TNum, self.model.agentsNum)
        self.totalPointX = totalPointX.values.reshape(TNum, self.model.agentsNum, 2)
        self.totalPointTheta = totalPointTheta.values.reshape(TNum, self.model.agentsNum)

        self.transient_index = int(0.9*self.totalPositionX.shape[0])
        self.tranPosition_x = self.totalPositionX[:, :, 0][self.transient_index:-1]
        self.tranPosition_y = self.totalPositionX[:, :, 1][self.transient_index:-1]
        self.tranPhaseTheta = self.totalPhaseTheta[self.transient_index:-1]


    @property
    def LastPositionX(self) -> np.ndarray:
        return self.totalPositionX[self.lookIndex]
    
    @property
    def LastPointX(self) -> np.ndarray:
        return self.totalPointX[self.lookIndex]
    
    @property
    def LastpPintTheta(self) -> np.ndarray:
        return self.totalPointTheta[self.lookIndex]
    
    @property
    def LastPhaseTheta(self) -> np.ndarray:
        return self.totalPhaseTheta[self.lookIndex]
    

    def draw_mp4(self,savePath: str = None, mp4Path: str = "./mp4",step: int = 1):
        if savePath is None:
            print("Please input the savePath")
        maxAbsPos = np.max(np.abs(self.totalPositionX)) 
        def plot_frame(i):
            pbar.update(1)
            positionX = self.totalPositionX[i]
            phaseTheta = self.totalPhaseTheta[i]
            pointX = self.totalPointX[i]
            pointTheta = self.totalPointTheta[i]
            fig.clear() # Clear the previous frame
            ax1 = plt.subplot(1, 1, 1)

            # dx = pointX[:, 0]
            # dy = pointX[:, 1]

            # magnitudes = np.sqrt(dx**2 + dy**2)

            # magnitudes[magnitudes == 0] = 1

            # dx_normalized = dx / magnitudes
            # dy_normalized = dy / magnitudes

            # fixed_length = 0.1
            # dx_fixed = dx_normalized * fixed_length
            # dy_fixed = dy_normalized * fixed_length

            # qv = ax1.quiver(positionX[:, 0], positionX[:, 1], dx_fixed, dy_fixed, 
            #            phaseTheta, cmap='viridis', clim=(0, 2*np.pi))

            # qv.set_clim(0, 2*np.pi)
            scatter = ax1.scatter(positionX[:, 0], positionX[:, 1], c=phaseTheta, cmap='viridis', clim=(0, 2*np.pi))
            scatter.set_clim(0, 2*np.pi)

            ax1.set_xlim(-1.1*maxAbsPos, 1.1*maxAbsPos) 
            ax1.set_ylim(-1.1*maxAbsPos, 1.1*maxAbsPos) 
            roundBound = np.round(maxAbsPos) # 四舍五入
            ax1.set_xticks([-roundBound, -roundBound / 2, 0, roundBound / 2, roundBound]) # 设置x轴刻度
            ax1.set_yticks([-roundBound, -roundBound / 2, 0, roundBound / 2, roundBound])   # 设置y轴刻度

        frames = np.arange(0, self.TNum, step) # 生成帧数
        pbar = tqdm(total=len(frames))
        fig, ax = plt.subplots(figsize=(5, 5))
        ani = ma.FuncAnimation(fig, plot_frame, frames=frames, interval=50, repeat=False)# 如何倍速播放，interval=50表示每50ms更新一次，想快点就设置小一点
        # 判断是否存在mp4文件夹，不存在则创建
        if not os.path.exists(mp4Path):
            os.makedirs(mp4Path)
        ani.save(f"{mp4Path}/{self.model}.mp4", dpi=200) # dpi是分辨率
        plt.close()

        pbar.close()

    @property
    def cal_S(self):
        phi = np.arctan2(self.tranPosition_y, self.tranPosition_x)
        W_plus = np.exp(1j*(phi + self.tranPhaseTheta))
        W_minus = np.exp(1j*(phi - self.tranPhaseTheta))

        time_OP_plus = np.abs(W_plus.mean(axis=1))
        time_OP_minus = np.abs(W_minus.mean(axis=1))
        S_plus = time_OP_plus.mean()
        S_minus = time_OP_minus.mean()

        S = max(S_plus, S_minus)
        return S
    
    @property
    def cal_R(self):
        Z = np.exp(1j*self.tranPhaseTheta)
        time_R = np.abs(Z.mean(axis=1))
        R = time_R.mean()
        return R
    
    @property
    def cal_V(self):
        V = np.mean(np.sqrt(self.tranPosition_x**2 + self.tranPosition_y**2))
        return V

    @property
    def cal_local_r(self):
        last_tranPosition_x = np.mean(self.tranPosition_x,axis=0)
        last_tranPosition_y = np.mean(self.tranPosition_y,axis=0)
        last_tranPhaseTheta = np.mean(self.tranPhaseTheta,axis=0)
        phi_last = np.arctan2(last_tranPosition_y, last_tranPosition_x)
        correlation =  last_tranPhaseTheta - phi_last
        cor_matrix = correlation - correlation[:, np.newaxis]
        r_ij = np.sqrt((1+np.cos(cor_matrix))/2)
        k_ij = np.mean(r_ij, axis=0)
        return k_ij
    
    # @property
    def find_gamma(self):
        tolerance = 0.1
        gamma = 0
        phi = np.arctan2(self.tranPosition_y, self.tranPosition_x)
        for i in range(1, phi.shape[1]):
            y = phi[:,i]
            temp = np.sin(y)
            temp = (max (temp) - min(temp)) / 2.0
            if temp > 1-tolerance:
                gamma += 1
        return gamma/phi.shape[1]

