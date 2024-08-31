from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
from itertools import product, combinations_with_replacement, chain
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import linregress
from random import choices
from scipy.signal import find_peaks

sigma = 10
beta = 8/3
rho = 28

def plot_2d(df):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(df.tolist())
    plt.show()

def plot_3d(df):
    fig = plt.figure()
    min_ = df.min()[[z for z in range(df.shape[1]-1)]].tolist()
    max_ = df.max()[[z for z in range(df.shape[1]-1)]].tolist()
    ax = fig.add_subplot(projection='3d')
    xaxis = df[0].tolist()
    yaxis = df[1].tolist()
    zaxis = df[2].tolist()
    ax.plot(xaxis, yaxis, zaxis)
    ax.set(xlabel='x', ylabel='y', zlabel='z', xlim=(min_[0],max_[0]), ylim=(min_[1],max_[1]), zlim=(min_[2],max_[2]))

    plt.show()

def lorenz(t, y):
    return (sigma*(y[1] - y[0]), y[0]*(rho-y[2])-y[1], y[0]*y[1] - beta*y[2])

a = 0.2
b = 0.2
c = 5.7

def rossler(t, y):
    return -y[1] - y[2], y[0] + a*y[1], b + y[2]*(y[0] - c)

def lle_calc(df_forecast, df_lyapunov, print_graphs=False):
    dt = df_forecast.loc[1,'t'] - df_forecast.loc[0,'t']
    df_diff = np.log((df_forecast-df_lyapunov).pow(2).sum(axis=1).pow(0.5))/dt
    plt.plot(df_diff)
    plt.show()
    maximum_idx = df_diff.idxmax()
    minimum_idx = df_diff.idxmin()
    #df_diff_exp = df_diff[minimum_idx:maximum_idx]
    df_diff_exp = df_diff[2000:3600]
    #df_diff_exp = df_diff.copy()
    df_diff_exp = df_diff_exp.reset_index(drop=True)
    x, y = find_peaks(-df_diff_exp)
    diff_keep = df_diff_exp
    df_diff_exp = df_diff_exp[x]
    print(-df_diff_exp)
    print(x)
    lle = round(linregress(x=(df_diff_exp.index).tolist(), y=df_diff_exp.tolist()).slope, 3)

    if print_graphs:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot((df_diff_exp.index*dt).tolist(), df_diff_exp.tolist())
        x = np.array(df_diff_exp.index*dt)
        y = linregress(x=(df_diff_exp.index*dt).tolist(), y=df_diff_exp.tolist()).intercept + x*linregress(x=(df_diff_exp.index*dt).tolist(), y=df_diff_exp.tolist()).slope
        ax.plot(x, y)
        plt.show(block=True)

        '''
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot((df_diff.index*dt).tolist(), df_diff.tolist())
        x = np.array(df_diff_exp.index*dt)
        y = linregress(x=(df_diff.index*dt).tolist(), y=df_diff.tolist()).intercept + x*linregress(x=(df_diff.index*dt).tolist(), y=df_diff.tolist()).slope
        ax.plot(x, y)
        plt.show(block=True)
        '''

    print(f'Average LLE: {lle}')
    return lle
    
def new_solve(function):
    a = solve_ivp(function, [0,10000], [0.1,0,0], t_eval=np.arange(0,100,0.01))
    df1 = pd.DataFrame([a.t, a.y[0], a.y[1], a.y[2]]).T
    df1.columns = ['t', 0, 1, 2]

    b = solve_ivp(function, [0,10000], [0.1,0,1e-9], t_eval=np.arange(0,100,0.01))
    df2 = pd.DataFrame([b.t, b.y[0], b.y[1], b.y[2]]).T
    df2.columns = ['t', 0, 1, 2]

    df1.to_csv('df1.csv', index=False)
    df2.to_csv('df2.csv', index=False)

    return df1, df2

'''
df1 = pd.read_csv('df1.csv')
df2 = pd.read_csv('df2.csv')
df_diff = lle_calc(df1, df2, True)
'''

#df1, df2 = new_solve(rossler)

df1 = pd.read_csv('df1-lorenz.csv')
df2 = pd.read_csv('df2-lorenz.csv')
lle_calc(df1,df2,1)
