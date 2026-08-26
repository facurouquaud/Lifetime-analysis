# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:55:29 2026

@author: Luis1
"""

import struct

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
import read_PTU_pixels_2 as rd
import Analisis_lifetime as lf
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
def leer_fotones_globales_ptu(archivo_ptu, apd_canal=1):
    """
    Lee TODOS los fotones del canal 'apd_canal' (1 ó 2) de un archivo PTU en modo T3,
    devolviendo:
        dtime_array : array de microtiempos (bins TCSPC)
        truensync_array : array de macrotiempos (ticks truensync)
        globRes, timeRes : resoluciones global y de microtiempo (del header)
    No usa filtrar_pixels ni markers, sólo parsea el stream crudo.
    """
    T3WRAPAROUND = 65536
    ofl = 0

    with open(archivo_ptu, 'rb') as f:
        numRecords, globRes, timeRes = rd.readHeaders(f)

        dtime_list = []
        truensync_list = []

        for _ in range(numRecords):
            raw = f.read(4)
            if not raw:
                break
            recordData = struct.unpack('<I', raw)[0]

            channel = recordData >> 28
            dtime   = (recordData >> 16) & 0xFFF
            nsync   = recordData & 0xFFFF

            if channel == 15:
                # marker / overflow
                if dtime == 0:
                    ofl += T3WRAPAROUND
                # dtime != 0 → marker; lo ignoramos para lifetime global
            elif channel == apd_canal:
                # fotón del APD deseado
                truensync = ofl + nsync
                dtime_list.append(dtime)
                truensync_list.append(truensync)
            else:
                # otros canales: ignorar
                pass

    dtime_array     = np.array(dtime_list,     dtype=np.int32)
    truensync_array = np.array(truensync_list, dtype=np.int64)

    return dtime_array, truensync_array, globRes, timeRes
# Load atto647N data (lifetime global, nuevo esquema)
path = r"C:\Users\Luis1\Downloads"
file = "\\10-otra-8"   # sin .ptu extra

archivo = path + file + ".ptu"

# Leer sólo fotones globales del canal 1 (APD rojo)
dtime_array, truensync_array, globRes, timeRes = leer_fotones_globales_ptu(
    archivo_ptu=archivo,
    apd_canal=1  # APD 1
)

# Convertir microtiempos a ns
t_ns = dtime_array * timeRes * 1e9   # timeRes [s/bin] → ns

# Histograma global
T = 25.0  # rango de interés (ns), opcional
fig, ax = plt.subplots()
ax.hist(t_ns, bins=70, color="firebrick", label="640R")
ax.set_xlabel("Tiempo [ns]", fontsize=16)
ax.set_ylabel("Cuentas", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.legend()
ax.grid()
ax.set_xlim(0, 25)
plt.tight_layout()
plt.show()


t_mask = (t_ns > 10) & (t_ns < 25)
t_sel = t_ns[t_mask]
#%%
print(f"Number of photons in selected range: {len(t_sel)}")

# Check if we have enough data
if len(t_sel) < 8:  # Minimum threshold for meaningful fit
    print("Not enough photons in the selected time range for fitting")
else:
    bins = 70
    counts, edges = np.histogram(t_sel, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Avoid empty bins when fitting
    mask_fit = counts > 0
    t_fit = bin_centers[mask_fit]
    counts_fit = counts[mask_fit]
    
    print(f"Number of non-empty bins: {len(counts_fit)}")
    
    # Check if we have enough non-empty bins for fitting
    if len(counts_fit) < 5:  # Need at least a few points for a meaningful fit
        print("Not enough non-empty bins for fitting")
    else:
        # Define exp_decay function if not already defined
        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C
        
        # -------- fitting ----------
        try:
            p0 = [counts_fit.max(), 4, 10]  # initial conditions
            
            popt, pcov = curve_fit(
                exp_decay,
                t_fit,
                counts_fit,
                p0=p0,
                maxfev=5000  # Increase max iterations if needed
            )
            
            A_fit, tau_fit, C_fit = popt
            tau_err = np.sqrt(np.diag(pcov))[1]
            
            # -------- plot ----------
            plt.figure(figsize=(7, 5))
            
            plt.plot(t_fit, counts_fit, 'o', label='640R', color="firebrick")
            
            t_model = np.linspace(t_fit.min(), t_fit.max(), 400)
            plt.plot(t_model,
                     exp_decay(t_model, *popt),
                     '-',
                     label=f'Ajuste: τ = {tau_fit:.2f} ± {tau_err:.2f} ns',
                     color="slategray")
            
            plt.xlabel("Tiempo [ns]", fontsize=16)
            plt.ylabel("Cuentas", fontsize=16)
            plt.grid()
            plt.legend(fontsize  =15)
            plt.tight_layout()
            plt.show()
            
            print(f"Tiempo de vida: {tau_fit:.3f} ± {tau_err:.3f} ns")
            
        except Exception as e:
            print(f"Fitting failed: {e}")
            
            # Plot raw data anyway to see what we have
            plt.figure(figsize=(7, 5))
            plt.plot(t_fit, counts_fit, 'o', label='640R', color="r")
            plt.xlabel("Tiempo [ns]", fontsize=16)
            plt.ylabel("Cuentas", fontsize=16)
            plt.grid()
            plt.legend(fontsize = 18)
            plt.title("Raw data (fitting failed)")
            plt.tight_layout()
            plt.show()

