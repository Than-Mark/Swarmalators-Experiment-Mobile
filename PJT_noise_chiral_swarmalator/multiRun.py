import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
from itertools import product
import seaborn as sns

from main import *

SAVE_PATH = r"./data"

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'

sns.set_theme(
    style="ticks", 
    font_scale=1.1, rc={
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


def runModel(model: ChiralActiveMatterWithNoise):
    # 运行模型
    model.run(60000)


if __name__ == "__main__":
    from multiprocessing import Pool, cpu_count

    # 这是扫描的噪声强度范围, 相当于图片的横坐标
    alphaRanges = [0, 0.5, 1.0, 1.5, 2.0]  # 空间噪声强度 alpha 的参数范围
    betaRanges = [0, 0.5, 1.0, 1.5, 2.0]   # 相位噪声强度 beta 的参数范围

    # 这里耦合强度和耦合距离是固定的, 只改变噪声强度, 如果需要尝试别的参数, 可以在这里手动修改
    models = [
        ChiralActiveMatterWithNoise(
            strengthLambda=0.95,   # 耦合强度 lambda
            distanceD0=2,          # 耦合距离 D0
            noiseRateAlpha=alpha,  # 空间噪声强度 alpha
            noiseRateBeta=beta,    # 相位噪声强度 beta
            savePath=SAVE_PATH,    # 保存路径
            overWrite=True,        # 是否覆盖已有缓存
            tqdm=True,             # 是否显示进度条
        )
        for alpha in alphaRanges
        for beta in betaRanges
    ]
    
    # 使用多进程加速计算
    with Pool(processes=cpu_count() - 2) as pool:
        results = list(tqdm(pool.imap(runModel, models), total=len(models)))

    pool.close()