# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 14:04:15 2025

@author: Luis1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
import read_PTU_pixels_2 as rd
plt.style.use(r"C:\Users\Luis1\Downloads\gula_style.mplstyle")
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
def filtrar_pixeles(pixeles, lower_limit, upper_limit):
    pixeles_filtrados = []

    for p in pixeles:
        n = len(p)
        if lower_limit <= n < upper_limit:
            if n > 0:
                pixeles_filtrados.append(np.asarray(p))

    return pixeles_filtrados

def filtro_pixeles_global(pixeles, lower_limit, upper_limit):
    fotones = []

    for p in pixeles:
        if lower_limit <= len(p) < upper_limit:
            if len(p) > 0:
                fotones.append(np.asarray(p))

    if len(fotones) == 0:
        return np.empty((0, 2))

    return np.vstack(fotones)


#%%
archivo = "C:\\Users\\Luis1\\Downloads\\Lifetime\\5x5um-50px-005ms.ptu"
archivo_2 = "C:\\Users\\Luis1\\Downloads\\Lifetime\\10x10um-200px-001ms.ptu"

with open(archivo, 'rb') as fd:
    numRecords, globRes, timeRes = rd.readHeaders(fd)
    dtime_array, truensync_array, pixeles = rd.readPT3_fast_pixels(fd, numRecords)

pixeles_validos = filtrar_pixeles(
    pixeles,
    lower_limit=15,
    upper_limit=100
)
datos = pixeles_validos[40]
plt.figure(constrained_layout=True)

plt.hist(datos[:, 0]* timeRes * 1e9, bins=14)
plt.xlabel("t / ns")
plt.ylabel("Counts")
plt.grid()
plt.show()



#%% Grafico global
datos_global = filtro_pixeles_global(pixeles, lower_limit=6, upper_limit=1000)
t_ns = datos_global[:, 0] * timeRes * 1e9

mask = (t_ns >= 9) & (t_ns <= 25)

datos_global_25ns = datos_global[mask]
plt.figure(constrained_layout=True)

plt.hist(datos_global_25ns[:, 0]* timeRes * 1e9, bins=20)
plt.xlabel("t / ns")
plt.ylabel("Counts")

plt.grid()
plt.show()


#%% Ajustes 
# tiempos en ns
t_ns = datos_global_25ns[:, 0] * timeRes * 1e9

bins = 40
hist, edges = np.histogram(t_ns, bins=bins)

t_centers = 0.5 * (edges[:-1] + edges[1:])
mask = hist >= 10

t_fit = t_centers[mask]
hist_fit = hist[mask]
def exponencial(t, A, tau, t0):
    return A * np.exp(-(t - t0)  / tau )
params, cov = curve_fit(
    exponencial,
    t_centers,
    hist,p0=(hist.max(), 8.0, 9.0), bounds=(
        (0, 0.1, 0.0),   # A, tau, t0 mínimos
        (np.inf, 20.0, 25.0)  # máximos físicos
    ))

A_fit, tau_fit, t0 = params
plt.figure(constrained_layout=True)
plt.hist(t_ns, bins=bins, alpha=0.7, label="Datos")
plt.title("Ajuste sin deconvolución")
plt.plot(
    t_centers,
    exponencial(t_centers, *params),
    'r-',
    label=f"Ajuste: τ = {tau_fit:.2f} ns")

plt.xlabel("t / ns")
plt.ylabel("Counts")
plt.legend()
plt.grid()
plt.show()
#%% Veamos como tratar la irf

tau_true = 3.0        # ns
t0 = 9              # ns (posición del pulso)
sigma_irf = 0.25      # ns (ancho IRF)
Nfot = 50_000
dt = 0.02             # ns
t = np.arange(0, 30, dt)

def irf(t, t0, sigma):
    return np.exp(-(t - t0)**2 / (2 * sigma**2))
IRF = irf(t, t0, sigma_irf)
IRF /= IRF.sum()   # normalización
decay = np.zeros_like(t)
mask = t >= t0
decay[mask] = np.exp(-(t[mask] - t0) / tau_true)
decay /= decay.sum()
signal = fftconvolve(decay, IRF, mode="same")
signal /= signal.sum()
cdf = np.cumsum(signal)
cdf /= cdf[-1]

rnd = np.random.rand(Nfot)
t_events = np.interp(rnd, cdf, t)
plt.plot(t, signal, label="Señal medida (IRF * exp)")
plt.plot(t, IRF / IRF.max() * signal.max(), '--', label="IRF (esc.)")
plt.xlabel("t / ns")
plt.ylabel("Prob.")
plt.legend()
plt.grid()
plt.show()
