import numpy as np
import pandas as pd
from itertools import product, combinations_with_replacement, chain
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import linregress
from random import choices

def indices(len_array, p):
    return list(combinations_with_replacement([x for x in range(0, len_array)], p))

def prod_mono(array, index):
    res = 1

    for i in index:
        res *= array[i]

    return res

def dat_parse(file):        # Data Parsing
    if file.split('.')[-1] == 'dat':
        df = pd.read_csv(file, sep=' ', header=None)

    else:
        df = pd.read_csv(file)

    df.columns = [x for x in range(len(df.columns))]
    df.rename(columns={0: 't'}, inplace=True)
    df.rename(columns={x: x-1 for x in range(len(df.columns))}, inplace=True)

    for c in df.columns:
        if c != 't':
            df[c] = df[c]/df.abs().max()[1:].max()

    return df

def ngrc(file, alpha=1e-3, n_ini=50000, k=2, s=1, n_learn=500, n_forecast=2000, order=[0,1,2], constant=1):
    args = locals()
    df = dat_parse(file)
    dt = df.loc[1,'t'] - df.loc[0,'t']
    d_in, d_out = len(df.columns)- 1, len(df.columns) - 1

    print(args)

    ### Training ###
    
    # O_total
    o_total = []
    df_learn = df.loc[n_ini - s*(k-1):n_ini + n_learn]
    
    for i in range(n_ini, n_ini + n_learn):
        line = df_learn.loc[i-s*k+1:i:s, [z for z in range(d_in)]]
        vars_in = list(chain(*line.values.tolist()))
        o_total_col = []

        for p in order:
            if p == 0:
                o_total_col.append(constant)
            else:
                for j in indices(len(vars_in), p):
                    o_total_col.append(prod_mono(vars_in, j))
            
        o_total.append(o_total_col)

    o_total = np.matrix(o_total).T

    # Y_d
    df1 = df_learn.diff().tail(n_learn)
    y_d = np.matrix(df1[[x for x in range(d_in)]]).T

    # W_out
    w_out = y_d @ o_total.T @ np.linalg.pinv((o_total @ o_total.T) + alpha*np.identity(np.shape(o_total)[0]))

    ### Learning Error Calculation ###

    df_forecast = df_learn.head(s*(k-1)+s).reset_index(drop=True)
    df_real = df_learn.tail(n_learn).reset_index(drop=True)
    y_forecast = np.matrix(df_forecast.tail(1)[[z for z in range(d_in)]]).T
   
    for i in range(n_learn):
        line = df_forecast.loc[i:i+s*(k-1):s]
        vars_in = list(chain(*line.loc[::, [z for z in range(d_in)]].values.tolist()))
        o_total_col = []
                
        for p in order:
            if p == 0:
                o_total_col.append(constant)
            else:
                for j in indices(len(vars_in), p):
                    o_total_col.append(prod_mono(vars_in, j))

        o_total_col = np.matrix(o_total_col).T
        y_forecast += w_out @ o_total_col
        
        x = pd.DataFrame(y_forecast.T)
        x['t'] = df_forecast.iloc[-1]['t'] + dt
        df_forecast = pd.concat([df_forecast, x], ignore_index=True)

    df_forecast = df_forecast.tail(n_learn).reset_index(drop=True)

    print('Erro do Aprendizado:', round(((df_forecast-df_real)[[z for z in range(d_in)]].pow(2).mean(axis=None))**.5, 5))

    ### Forecast ###

    df_forecast = df_learn.tail(s*(k-1)+s).reset_index(drop=True)
    df_real = df.loc[n_ini + n_learn + 1:n_ini + n_learn + n_forecast].reset_index(drop=True)
    y_forecast = np.matrix(df_forecast.tail(1)[[z for z in range(d_in)]]).T
   
    for i in range(n_forecast):
        line = df_forecast.loc[i:i+s*(k-1):s]
        vars_in = list(chain(*line.loc[::, [z for z in range(d_in)]].values.tolist()))
        o_total_col = []
                
        for p in order:
            if p == 0:
                o_total_col.append(constant)
            else:
                for j in indices(len(vars_in), p):
                    o_total_col.append(prod_mono(vars_in, j))

        o_total_col = np.matrix(o_total_col).T
        y_forecast += w_out @ o_total_col
        
        x = pd.DataFrame(y_forecast.T)
        x['t'] = df_forecast.iloc[-1]['t'] + dt
        df_forecast = pd.concat([df_forecast, x], ignore_index=True)

    df_forecast = df_forecast.tail(n_forecast).reset_index(drop=True)

    print('Erro da Previsão:', round(((df_forecast-df_real)[[z for z in range(d_in)]].pow(2).mean(axis=None))**.5, 5))

    return df_forecast, df_real

def ngrc_training(file, alpha=1e-3, n_ini=50000, k=2, s=1, n_learn=500, n_forecast=2000, order=[0,1,2], constant=1):
    args = locals()
    df = dat_parse(file)
    dt = df.loc[1,'t'] - df.loc[0,'t']
    d_in, d_out = len(df.columns)- 1, len(df.columns) - 1

    ### Training ###
    
    # o_total
    o_total = []
    df_learn = df.loc[n_ini - s*(k-1):n_ini + n_learn]
    
    for i in range(n_ini, n_ini + n_learn):
        line = df_learn.loc[i-s*k+1:i:s, [z for z in range(d_in)]]
        vars_in = list(chain(*line.values.tolist()))
        o_total_col = []

        for p in order:
            if p == 0:
                o_total_col.append(constant)
            else:
                for j in indices(len(vars_in), p):
                    o_total_col.append(prod_mono(vars_in, j))
            
        o_total.append(o_total_col)

    o_total = np.matrix(o_total).T

    # Y_d
    df1 = df_learn.diff().tail(n_learn)
    y_d = np.matrix(df1[[x for x in range(d_in)]]).T

    # W_out
    w_out = y_d @ o_total.T @ np.linalg.pinv((o_total @ o_total.T) + alpha*np.identity(np.shape(o_total)[0]))

    return list(args.values()), w_out

def ngrc_forecast(w_out, file, alpha=1e-3, n_ini=50000, k=2, s=1, n_learn=500, n_forecast=2000, order=[0,1,2], constant=1, lyapunov=False, epsilon=1e-9):
    df = dat_parse(file)
    dt = df.loc[1,'t'] - df.loc[0,'t']
    d_in, d_out = len(df.columns)- 1, len(df.columns) - 1
    df_learn = df.loc[n_ini - s*(k-1):n_ini + n_learn]
    df_forecast = df_learn.tail(s*(k-1)+s).reset_index(drop=True)
    df_real = df.loc[n_ini + n_learn + 1:n_ini + n_learn + n_forecast].reset_index(drop=True)

    if lyapunov:
        df_forecast.iloc[-1][0] += epsilon #DIFFERENCE

    y_forecast = np.matrix(df_forecast.tail(1)[[z for z in range(d_in)]]).T
   
    for i in range(n_forecast):
        line = df_forecast.loc[i:i+s*(k-1):s]
        vars_in = list(chain(*line.loc[::, [z for z in range(d_in)]].values.tolist()))
        o_total_col = []
                
        for p in order:
            if p == 0:
                o_total_col.append(constant)
            else:
                for j in indices(len(vars_in), p):
                    o_total_col.append(prod_mono(vars_in, j))

        o_total_col = np.matrix(o_total_col).T
        y_forecast += w_out @ o_total_col
        
        x = pd.DataFrame(y_forecast.T)
        x['t'] = df_forecast.iloc[-1]['t'] + dt
        df_forecast = pd.concat([df_forecast, x], ignore_index=True)

    df_forecast = df_forecast.tail(n_forecast).reset_index(drop=True)

    print('Erro da Previsão:', round(((df_forecast-df_real)[[z for z in range(d_in)]].pow(2).mean(axis=None))**.5, 5))

    return df_forecast, df_real
'''
def lle_calc(df_forecast, df_lyapunov, print_graphs=False):
    dt = df_forecast.loc[1,'t'] - df_forecast.loc[0,'t']
    df_diff = np.log((df_forecast-df_lyapunov).pow(2).sum(axis=1))/2
    lle = round(linregress(x=(df_diff.index*dt).tolist(), y=df_diff.tolist()).slope, 3)

    if print_graphs:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot((df_diff.index*dt).tolist(), df_diff.tolist())
        x = np.array(df_diff.index*dt)
        y = linregress(x=(df_diff.index*dt).tolist(), y=df_diff.tolist()).intercept + x*linregress(x=(df_diff.index*dt).tolist(), y=df_diff.tolist()).slope
        ax.plot(x, y)
        plt.show(block=True)

    print(f'Average LLE: {lle}')
    
    return lle
'''
def lle_calc(df_forecast, df_lyapunov, print_graphs=False):
    dt = df_forecast.loc[1,'t'] - df_forecast.loc[0,'t']
    df_diff = np.log((df_forecast-df_lyapunov).pow(2).sum(axis=1).pow(0.5))
    maximum_idx = df_diff.idxmax()
    df_diff_exp = df_diff[:maximum_idx]
    lle = round(linregress(x=(df_diff_exp.index*dt).tolist(), y=df_diff_exp.tolist()).slope, 3)

    if print_graphs:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot((df_diff_exp.index*dt).tolist(), df_diff_exp.tolist())
        x = np.array(df_diff_exp.index*dt)
        y = linregress(x=(df_diff_exp.index*dt).tolist(), y=df_diff_exp.tolist()).intercept + x*linregress(x=(df_diff_exp.index*dt).tolist(), y=df_diff_exp.tolist()).slope
        ax.plot(x, y)
        plt.show(block=True)

    print(f'Average LLE: {lle}')
    return lle

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

def plots_compare_2d(df_forecast, df_real):
    fig = plt.figure(figsize=plt.figaspect(.5))
    xaxis = df_forecast['t'].tolist()
    yaxis = df_forecast[0].tolist()
    ax = fig.add_subplot(1,2,1)
    ax.plot(xaxis, yaxis)
    plt.title('Forecast')
    
    xaxis2 = df_real['t'].tolist()
    yaxis2 = df_real[0].tolist()
    ax = fig.add_subplot(1,2,2)
    ax.plot(xaxis2, yaxis2)
    plt.title('Real Data')

    plt.show()

def plots_compare_3d(df_forecast, df_real):
    df = pd.concat([df_forecast, df_real])
    min_ = df.min()[[z for z in range(df.shape[1]-1)]].tolist()
    max_ = df.max()[[z for z in range(df.shape[1]-1)]].tolist()
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    xaxis = df_forecast[0].tolist()
    yaxis = df_forecast[1].tolist()
    zaxis = df_forecast[2].tolist()
    ax.plot(xaxis, yaxis, zaxis)
    ax.set(xlabel='x', ylabel='y', zlabel='z', xlim=(min_[0],max_[0]), ylim=(min_[1],max_[1]), zlim=(min_[2],max_[2]))
    plt.title('Forecast')
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    xaxis2 = df_real[0].tolist()
    yaxis2 = df_real[1].tolist()
    zaxis2 = df_real[2].tolist()
    ax.plot(xaxis2, yaxis2, zaxis2)
    ax.set(xlabel='x', ylabel='y', zlabel='z', xlim=(min_[0],max_[0]), ylim=(min_[1],max_[1]), zlim=(min_[2],max_[2]))
    plt.title('Real Data')

    plt.show()

def animation_plot(df, step=10, save=False):    #Animação Individual
    def update(frame):
        x = xaxis[:frame]
        y = xaxis[:frame]
        z = zaxis[:frame]
        scat._offsets3d = x, y, z

        return scat

    min_ = df.min()[[z for z in range(df.shape[1]-1)]].tolist()
    max_ = df.max()[[z for z in range(df.shape[1]-1)]].tolist()    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xaxis = df[0].tolist()
    yaxis = df[1].tolist()
    zaxis = df[2].tolist()
    scat = ax.scatter(xaxis[0], yaxis[0], zaxis[0], marker='.')
    ax.set(xlabel='x', ylabel='y', zlabel='z', xlim=(min_[0],max_[0]), ylim=(min_[1],max_[1]), zlim=(min_[2],max_[2]))

    ani = animation.FuncAnimation(fig, update, frames=range(0, df.shape[0], step), repeat=False)
    plt.show()

    if save:        
        writervideo = animation.PillowWriter(fps=30)
        ani.save('plot.gif', writer=writervideo)

def animation_plot_simultaneous(df1, df2, step=10): #Animação Simultânea
    def update(frame):
        x = xaxis[:frame]
        y = xaxis[:frame]
        z = zaxis[:frame]
        scat._offsets3d = x, y, z

        return scat

    def update2(frame):
        x = xaxis2[:frame]
        y = xaxis2[:frame]
        z = zaxis2[:frame]
        scat2._offsets3d = x, y, z

        return scat2

    df = pd.concat([df1,df2])
    min_ = df.min()[[z for z in range(df.shape[1]-1)]].tolist()
    max_ = df.max()[[z for z in range(df.shape[1]-1)]].tolist()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xaxis = df1[0].tolist()
    yaxis = df1[1].tolist()
    zaxis = df1[2].tolist()
    scat = ax.scatter(xaxis[0], yaxis[0], zaxis[0], marker='.')
    ax.set(xlabel='x', ylabel='y', zlabel='z', xlim=(min_[0],max_[0]), ylim=(min_[1],max_[1]), zlim=(min_[2],max_[2]))
    plt.title('Real Data')

    ani = animation.FuncAnimation(fig, update, frames=range(0, df1.shape[0], step), repeat=False)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    xaxis2 = df2[0].tolist()
    yaxis2 = df2[1].tolist()
    zaxis2 = df2[2].tolist()
    scat2 = ax2.scatter(xaxis2[0], yaxis2[0], zaxis2[0], marker='.')
    ax2.set(xlabel='x', ylabel='y', zlabel='z', xlim=(min_[0],max_[0]), ylim=(min_[1],max_[1]), zlim=(min_[2],max_[2]))
    plt.title('Forecast Data')

    ani2 = animation.FuncAnimation(fig2, update2, frames=range(0, df2.shape[0], step), repeat=False)

    plt.show()

df = pd.read_csv('inmet-full.csv')

