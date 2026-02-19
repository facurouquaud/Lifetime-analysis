# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:00:24 2026

@author: Luis1
"""
import struct
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import read_PTU_pixels_2 as rd
#plt.style.use(r"C:\Users\Luis1\Downloads\gula_style.mplstyle")
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
import matplotvanda as vd
archivo_r = "C:\\Users\\Luis1\\Downloads\\pulsos_int_rojo.ptu"
archivo_a = "C:\\Users\\Luis1\\Downloads\\pulsos_int_am.ptu"

with open(archivo_r, 'rb') as fd:
    numRecords, glob, timer = rd.readHeaders(fd)
    dtime, truesync, pixeles_r = rd.readPT3_fast_pixels(fd, numRecords)
with open(archivo_a, 'rb') as fd:
    numRecords, glob, timer = rd.readHeaders(fd)
    dtime, truesync, pixeles_a = rd.readPT3_fast_pixels(fd, numRecords)    


dtime_apd1 = np.concatenate(
    [pixel_dict[1][:,0] 
     for pixel_dict in pixeles_r 
     if pixel_dict[1].size > 0]
)
dtime_apd2 = np.concatenate(
    [pixel_dict[1][:,0] 
     for pixel_dict in pixeles_a 
     if pixel_dict[1].size > 0]
)


fig,ax = plt.subplots()
ax.hist(dtime_apd1*timer*1E9, bins=850, density = True, color = "darkred", label = "Excitación rojo")
ax.hist(dtime_apd2*timer*1E9, bins = 850,density = True, color = "darkorange", label = "Excitación amarillo")
ax.set_xlabel("Tiempo [ns]", fontsize = 14)
ax.set_ylabel("Densidad de probabilidad", fontsize = 14)
vd.gula_grid(ax)
plt.tight_layout()
ax.legend()
plt.show()



#%% Hallamos distancia
from scipy.signal import find_peaks

# Convertimos a tiempo real
tiempos_r = dtime_apd1 * timer*1E9
tiempos_a = dtime_apd2 * timer*1E9

# Histogramas
bins = 1000

hist_r, edges_r = np.histogram(tiempos_r, bins=bins, density=True)
hist_a, edges_a = np.histogram(tiempos_a, bins=bins, density=True)

# Centros de bin
centros_r = 0.5 * (edges_r[:-1] + edges_r[1:])
centros_a = 0.5 * (edges_a[:-1] + edges_a[1:])
peaks_r, _ = find_peaks(hist_r, height=np.max(hist_r)*0.3)
peaks_a, _ = find_peaks(hist_a, height=np.max(hist_a)*0.3)
# Pico principal = el más alto
pico_principal_r = centros_r[peaks_r[np.argmax(hist_r[peaks_r])]]
pico_principal_a = centros_a[peaks_a[np.argmax(hist_a[peaks_a])]]

print(f"Pico rojo: {pico_principal_r:.2f} ns")
print(f"Pico amarillo: {pico_principal_a:.2f} ns")
distancia = abs(pico_principal_r - pico_principal_a)
print(f"Separación entre pulsos: {distancia:.2f} ns")
fig, ax = plt.subplots()

ax.plot(centros_r, hist_r, color="darkred", label="Excitación rojo")
ax.plot(centros_a, hist_a, color="darkorange", label="Excitación amarillo")



ax.set_xlabel("Tiempo [ns]")
ax.set_ylabel("Densidad de probabilidad")
vd.gula_grid(ax)
ax.legend()

plt.tight_layout()
plt.show()

sep_1 = 24.93 
#%%

