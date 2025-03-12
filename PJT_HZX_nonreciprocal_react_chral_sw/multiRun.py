# 这是一个初始化文件，用于导入常用的库和设置一些全局的参数
import matplotlib
matplotlib.use('agg')
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
from main import ChiralActiveMatterNonreciprocalReact
from main import draw_mp4
from main import plot_last
import logging

randomSeed = 10

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
)

@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

import seaborn as sns
import matplotlib.font_manager as fm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

param_sets = [
    {'Lambdas': [0.01], 'D0Means': [0.1], 'D0Stds': [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]},
    {'Lambdas': [0.01], 'D0Means': [0.25], 'D0Stds': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
    {'Lambdas': [0.01], 'D0Means': [0.5], 'D0Stds': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
    {'Lambdas': [0.01], 'D0Means': [0.75], 'D0Stds': [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5]},
    {'Lambdas': [0.01], 'D0Means': [1], 'D0Stds': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]},
    {'Lambdas': [0.025], 'D0Means': [0.1], 'D0Stds': [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]},
    {'Lambdas': [0.025], 'D0Means': [0.25], 'D0Stds': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
    {'Lambdas': [0.025], 'D0Means': [0.5], 'D0Stds': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
    {'Lambdas': [0.025], 'D0Means': [0.75], 'D0Stds': [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5]},
    {'Lambdas': [0.025], 'D0Means': [1], 'D0Stds': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]},
    # {'Lambdas': [0.05], 'D0Means': [0.1], 'D0Stds': [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]},
    # {'Lambdas': [0.05], 'D0Means': [0.25], 'D0Stds': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
    # {'Lambdas': [0.05], 'D0Means': [0.5], 'D0Stds': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
    # {'Lambdas': [0.05], 'D0Means': [0.75], 'D0Stds': [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5]},
    # {'Lambdas': [0.05], 'D0Means': [1], 'D0Stds': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]},
    # {'Lambdas': [0.075], 'D0Means': [0.1], 'D0Stds': [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]},
    # {'Lambdas': [0.075], 'D0Means': [0.25], 'D0Stds': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
    # {'Lambdas': [0.075], 'D0Means': [0.5], 'D0Stds': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
    # {'Lambdas': [0.075], 'D0Means': [0.75], 'D0Stds': [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5]},
    # {'Lambdas': [0.075], 'D0Means': [1], 'D0Stds': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]},
    # {'Lambdas': [0.1], 'D0Means': [0.1], 'D0Stds': [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]},
    # {'Lambdas': [0.1], 'D0Means': [0.25], 'D0Stds': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
    # {'Lambdas': [0.1], 'D0Means': [0.5], 'D0Stds': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
    # {'Lambdas': [0.1], 'D0Means': [0.75], 'D0Stds': [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5]},
    # {'Lambdas': [0.1], 'D0Means': [1], 'D0Stds': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]},
    # {'Lambdas': [0.2], 'D0Means': [0.1], 'D0Stds': [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]},
    # {'Lambdas': [0.2], 'D0Means': [0.25], 'D0Stds': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
    # {'Lambdas': [0.2], 'D0Means': [0.5], 'D0Stds': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
    # {'Lambdas': [0.2], 'D0Means': [0.75], 'D0Stds': [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5]},
    # {'Lambdas': [0.2], 'D0Means': [1], 'D0Stds': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]},
    # {'Lambdas': [0.3], 'D0Means': [0.1], 'D0Stds': [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]},
    # {'Lambdas': [0.3], 'D0Means': [0.25], 'D0Stds': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
    # {'Lambdas': [0.3], 'D0Means': [0.5], 'D0Stds': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
    # {'Lambdas': [0.3], 'D0Means': [0.75], 'D0Stds': [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5]},
    # {'Lambdas': [0.3], 'D0Means': [1], 'D0Stds': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]},
]

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['animation.ffmpeg_path'] = "D:/ffmpeg/bin/ffmpeg.exe"

def run_model(model):
    try:
        model.run(60000)
        draw_mp4(model)
        # model.save()
        logging.info(f"Model with Lambda={model.strengthLambda}, D0Mean={model.distanceD0Mean}, D0Std={model.distanceD0Std} completed successfully")
    except Exception as e:
        logging.error(f"Error running model: {e}")

if __name__ == "__main__":
      
    sns.set(font_scale=1.1, rc={
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

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['animation.ffmpeg_path'] = "D:/ffmpeg/bin/ffmpeg.exe"


    models = []
    for param_set in param_sets:
        for Lambda, D0Mean, D0Std in product(param_set['Lambdas'], param_set['D0Means'], param_set['D0Stds']):
            model = ChiralActiveMatterNonreciprocalReact(
                chiralNum=1, 
                strengthLambda=Lambda, 
                distanceD0Mean=D0Mean, 
                distanceD0Std=D0Std, 
                d0Distribution='uniform', 
                omegaDistribution='uniform', 
                tqdm=True, 
                savePath="./data/", 
                overWrite=True
            )
            models.append(model)

    with Pool(processes=12) as p:
        p.map(run_model, models)




# import matplotlib.colors as mcolors
# import matplotlib.animation as ma
# import matplotlib.pyplot as plt
# from multiprocessing import Pool
# from tqdm import tqdm
# from itertools import product
# import pandas as pd
# import numpy as np
# import numba as nb
# import imageio
# import sys
# import os
# import shutil
# from main import ChiralActiveMatterNonreciprocalReact
# # from main import draw_mp4

# randomSeed = 10

# # %matplotlib inline
# # %config InlineBackend.figure_format = "retina"

# new_cmap = mcolors.LinearSegmentedColormap.from_list(
#     "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
# )

# @nb.njit
# def colors_idx(phaseTheta):
#     return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

# import seaborn as sns
# import matplotlib.font_manager as fm

# def run_model(model):
#         model.run(60000)

# if __name__ == "__main__":
      
#     sns.set(font_scale=1.1, rc={
#         'figure.figsize': (6, 5),
#         'axes.facecolor': 'white',
#         'figure.facecolor': 'white',
#         'grid.color': '#dddddd',
#         'grid.linewidth': 0.5,
#         "lines.linewidth": 1.5,
#         'text.color': '#000000',
#         'figure.titleweight': "bold",
#         'xtick.color': '#000000',
#         'ytick.color': '#000000'
#     })

#     plt.rcParams['mathtext.fontset'] = 'stix'
#     plt.rcParams['font.family'] = 'STIXGeneral'
#     # plt.rcParams['animation.ffmpeg_path'] = "D:/ffmpeg/bin/ffmpeg.exe"

# # 单手性，d0均匀分布宽度0.1
#     Lambdas = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
#     D0Means_range = np.arange(0.05, 1.05, 0.05)
#     D0Means = np.append(D0Means_range, 2)
#     # ODs = {'uniform', 'normal', 'lorentzian'}

#     models = [
#         ChiralActiveMatterNonreciprocalReact(chiralNum = 1, 
#                                              strengthLambda = Lambda, 
#                                              distanceD0Mean = D0Mean, distanceD0Std=0.1, d0Distribution = 'uniform', 
#                                              omegaDistribution = 'uniform', 
#                                              tqdm=True, savePath="./data/", overWrite=True
#         )
#         for Lambda, D0Mean, in product(Lambdas, D0Means)
#     ]
#     with Pool(processes=16) as p:
#         p.map(run_model, models)





# import matplotlib.colors as mcolors
# import matplotlib.animation as ma
# import matplotlib.pyplot as plt
# from multiprocessing import Pool
# from tqdm.notebook import tqdm
# from itertools import product
# import pandas as pd
# import numpy as np
# import numba as nb
# import imageio
# import sys
# import os
# import shutil

# randomSeed = 100

# new_cmap = mcolors.LinearSegmentedColormap.from_list(
#     "new", plt.cm.hsv(np.linspace(0, 1, 256)) * 0.85, N=256
# )

# @nb.njit
# def colors_idx(phaseTheta):
#     return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)

# import seaborn as sns
# import matplotlib.font_manager as fm

# def run_model(model):
#         model.run(10)


# if __name__ == "__main__":

#     sns.set_theme(font_scale=1.1, rc={
#         'figure.figsize': (6, 5),
#         'axes.facecolor': 'white',
#         'figure.facecolor': 'white',
#         'grid.color': '#dddddd',
#         'grid.linewidth': 0.5,
#         "lines.linewidth": 1.5,
#         'text.color': '#000000',
#         'figure.titleweight': "bold",
#         'xtick.color': '#000000',
#         'ytick.color': '#000000'
#     })
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.rcParams['mathtext.fontset'] = "cm"

#     from main import *

#     # 扫描的参数范围
#     Js = [0.5]
#     Ks = np.arange(-1, 0.21, 0.1).round(2)
#     d0s = [np.inf]

#     models = [
#         ShortRangePhaseInter(
#             K=K, J=J, d0=d0, 
#             tqdm=True, savePath="./data", overWrite=True
#         ) 
#         for J, K, d0 in product(Js, Ks, d0s)
#     ]

#     # processes为进程数，表示同时执行的进程数，可以根据CPU核数进行设置
#     with Pool(processes=4) as p:
#         p.map(run_model, models)