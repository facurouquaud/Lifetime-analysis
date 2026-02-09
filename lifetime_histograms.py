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

def filtrar_pixeles(pixeles, apd, lower_limit, upper_limit):
    pixeles_filtrados = []

    for p in pixeles:
        if apd not in p:
            continue

        n = p[apd].shape[0]   # nº de fotones en ese APD

        if lower_limit <= n < upper_limit:
            pixeles_filtrados.append(p)

    return pixeles_filtrados


def fotones_globales(pixeles, apd, lower_limit, upper_limit):
    fotones = []

    for p in pixeles:
        if apd not in p:
            continue

        n = p[apd].shape[0]

        if lower_limit <= n < upper_limit:
            fotones.append(p[apd])

    if len(fotones) == 0:
        return np.empty((0, 2))

    return np.vstack(fotones)


#%% Cargo todos los pixeles de las mediciones
mediciones = ["10x10-200px-30us","5x5-100px-60us","2x2-50px-400us" ]
datos = []
for i in range(len(mediciones)): 
 archivo = f"C:\\Users\\Luis1\\Downloads\\Mediciones_intercalados\\{mediciones[i]}\\ARLRLA.ptu"
 with open(archivo, 'rb') as fd:
        numRecords, globRes, timeRes = rd.readHeaders(fd)
        dtime_array, truensync_array, pixeles = rd.readPT3_fast_pixels(fd, numRecords)
        datos.append(pixeles)

pixeles_validos = filtrar_pixeles(
    pixeles,
    apd=1,              # APD 1
    lower_limit=5,
    upper_limit=1000
)


#%% Grafico lifetime global
colors = ["r", "y"]
labels = ["APD1 (R)", "APD2 (A)"]
regions = ["10x10 $\mu m$", "5x5 $\mu m$", "2x2 $\mu m$"]

for j in range(3):

    plt.figure(figsize=(6,4))

    for i in range(1, 3):  # APD 1 y 2

        datos_globales = fotones_globales(
            datos[j],          
            apd=i,
            lower_limit=1,
            upper_limit=200
        )

        t_ns = datos_globales[:, 0] * timeRes * 1e9

        plt.hist(
            t_ns,
            bins=200,
            density=True,
            color=colors[i-1],
            label=labels[i-1]
        )

    plt.xlabel("Tiempo [ns]")
    plt.ylabel("Densidad de probabilidad")
    plt.legend()
    plt.grid(True)

    plt.text(
        0.872, 0.87,
        regions[j],
        transform=plt.gca().transAxes,
        va='top', ha='left',
        fontsize=20,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()



    
#%% Histograma construido a mano (para calcular diferencia de tiempos)
from scipy.signal import find_peaks

counts, bin_edges = np.histogram(
    datos_globales[:, 0] * timeRes * 1e9,
    bins=200,
    density=True
)

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
peaks, properties = find_peaks(
    counts,
    height=0.05,      # umbral mínimo (ajustar)
    distance=5        # separación mínima entre picos (bins)
)
plt.plot(bin_centers, counts, color=colors[i-1], label=label[i-1])
plt.plot(bin_centers[peaks], counts[peaks], "x", color="black", ms=8)

plt.xlabel("t / ns")
plt.ylabel("Counts")
plt.legend()
plt.grid(True)
print(f"Tiempo entre eventos amarillos {bin_centers[peaks[1]] - bin_centers[peaks[0]]}")


#%% Ajustes (ver bien lo de la IRF) 
# tiempos en ns
t_ns = datos_globales[:, 0] * timeRes * 1e9

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
