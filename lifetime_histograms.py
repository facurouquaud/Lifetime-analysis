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
 archivo = f"C:\\Users\\Luis1\\Downloads\\Mediciones_intercalados\\{mediciones[i]}\\AALALR.ptu"
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
regions = ["10x10 $\mu m$"]


fig,ax = plt.subplots(figsize=(18,4))

for i in range(1, 3):  # APD 1 y 2

    datos_globales = fotones_globales(
        datos[0],          
        apd=i,
        lower_limit=1,
        upper_limit=200
    )

    t_ns = datos_globales[:, 0] * timeRes * 1e9

    ax.hist(
        t_ns,
        bins=200,
        density=True,
        color=colors[i-1],
        label=labels[i-1]
    )

ax.set_xlabel("Tiempo [ns]", fontsize = 14)
ax.set_ylabel("Densidad de probabilidad", fontsize = 14)
ax.legend()
ax.grid()
# plt.text(
#     0.1, 0.95,
#     regions[0],
#     transform=plt.gca().transAxes,
#     va='top', ha='left',
#     fontsize=14,
#     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
# )

plt.tight_layout()
plt.show()



#%% Reconstrucción imágenes por lifetime pixel apixel
archivo = "C:\\Users\\Lenovo\\Downloads\\Lifetime\\10x10-200px-30us\\AALRLA.ptu"
with open(archivo, 'rb') as fd:
       numRecords, globRes, timeRes = rd.readHeaders(fd)
       dtime_array, truensync_array, pixeles = rd.readPT3_fast_pixels(fd, numRecords)
       datos.append(pixeles)
pixeles, ida, vuelta = rd.filtrar_pixeles(pixeles,200,4,1)
ida = ida[3:len(ida) - 157]

#%%
def reconstruir_imagen_ventanas(pixeles, apd, timeRes, ventanas):

    n_total = len(pixeles)
    lado = int(np.sqrt(n_total))

    if lado * lado != n_total:
        raise ValueError("El número de píxeles no corresponde a una grilla cuadrada")

    imagenes = []

    for w in ventanas:
        img = np.zeros(n_total)

        for i, p in enumerate(pixeles):

            if apd not in p:
                continue

            if len(p[apd]) == 0:
                continue

            t_ns = p[apd] * timeRes * 1e9

            mask = (t_ns >= w[0]) & (t_ns < w[1])
            img[i] = np.sum(mask)

        imagenes.append(img.reshape(lado, lado))

    return imagenes
ventanas = [(0,25), (25,50), (50,75), (75,100)]
imgs = reconstruir_imagen_ventanas(
    ida,
    apd=1,
    timeRes=timeRes,
    ventanas=ventanas
)


# tamaño físico
L = 10  # micrómetros
n_pix = imgs[0].shape[0]
pixel_size = L / n_pix

extent = [0, L, 0, L]  # eje en micrómetros

vmax = max([img.max() for img in imgs])
fig, axes = plt.subplots(1, 4, figsize=(18,4))

for i, w in enumerate(ventanas):
    
    im = axes[i].imshow(imgs[i],
                        cmap='inferno',
                        vmin=0,
                        vmax=vmax,
                        extent=extent,
                        origin='lower')

    axes[i].set_title(f"{w[0]}–{w[1]} ns", fontsize = 16)
    
    axes[i].set_xlabel("x (µm)", fontsize = 16)
    axes[i].set_ylabel("y (µm)", fontsize = 16)
    axes[i].tick_params(axis='both', which='major', labelsize=13)

    
    axes[i].set_xticks(np.arange(0, L+1, 2))
    axes[i].set_yticks(np.arange(0, L+1, 2))

# Colorbar única
#cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
#cbar.set_label("Fotones")

plt.tight_layout()
plt.show()

#%% Ajuste lifetime 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#APD rojo

def exp_decay(t, A, tau, C):
    return A * np.exp(-t/tau) + C


apd = 1  

datos_globales = fotones_globales(
    datos[0],
    apd=apd,
    lower_limit=20,
    upper_limit=200
)

t_ns = datos_globales[:, 0] * timeRes * 1e9


mask = (t_ns >= 25) & (t_ns < 52)
t_sel = t_ns[mask]

t_sel = t_sel - 31


bins = 100
counts, edges = np.histogram(t_sel, bins=bins, range=(0,23))
bin_centers = (edges[:-1] + edges[1:]) / 2

# evitar bins vacíos al ajustar
mask_fit = counts > 0
t_fit = bin_centers[mask_fit]
counts_fit = counts[mask_fit]


# -------- ajuste ----------
p0 = [counts_fit.max(), 5, 80]  # condiciones iniciales

popt, pcov = curve_fit(
    exp_decay,
    t_fit,
    counts_fit,
    p0=p0
)

A_fit, tau_fit, C_fit = popt
tau_err = np.sqrt(np.diag(pcov))[1]


# -------- gráfico ----------
plt.figure(figsize=(7,5))

plt.plot(t_fit, counts_fit, 'o', label='Datos', color = "indianred")

t_model = np.linspace(0,22,400)
plt.plot(t_model,
         exp_decay(t_model, *popt),
         '-',
         label=f'Ajuste: τ = {tau_fit:.2f} ± {tau_err:.2f} ns', color = "slategray")

plt.xlabel("Tiempo [ns]", fontsize = 16)
plt.ylabel("Cuentas", fontsize = 16)
plt.grid()
plt.legend(fontsize = 13)
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=13)

plt.show()

print(f"Tiempo de vida: {tau_fit:.3f} ± {tau_err:.3f} ns")

#%%
#APD amarillo


apd = 2  

datos_globales = fotones_globales(
    datos[0],
    apd=apd,
    lower_limit=1,
    upper_limit=200
)
t_ns = datos_globales[:, 0] * timeRes * 1e9

t_sel = t_ns - 4.5

bins = 100
counts, edges = np.histogram(t_sel, bins=bins, range=(0,24))
bin_centers = (edges[:-1] + edges[1:]) / 2

# evitar bins vacíos al ajustar
mask_fit = counts > 0
t_fit = bin_centers[mask_fit]
counts_fit = counts[mask_fit]


# -------- ajuste ----------
p0 = [counts_fit.max(), 5, 1]  # condiciones iniciales

popt, pcov = curve_fit(
    exp_decay,
    t_fit,
    counts_fit,
    p0=p0
)

A_fit, tau_fit, C_fit = popt
tau_err = np.sqrt(np.diag(pcov))[1]


# -------- gráfico ----------
plt.figure(figsize=(7,5))

plt.plot(t_fit, counts_fit, 'o', label='Datos', color = "y")

t_model = np.linspace(0,22,400)
plt.plot(t_model,
         exp_decay(t_model, *popt),
         '-',
         label=f'Ajuste: τ = {tau_fit:.2f} ± {tau_err:.2f} ns', color = "slategray")

plt.xlabel("Tiempo [ns]", fontsize = "16")
plt.ylabel("Cuentas", fontsize = "16")
plt.grid()
plt.legend(fontsize = 13)
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=13)

plt.show()

print(f"Tiempo de vida: {tau_fit:.3f} ± {tau_err:.3f} ns")