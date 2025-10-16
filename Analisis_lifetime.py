# -*- coding: utf-8 -*-
"""
Archivo para estudiar las imágenes de lifetime, con datos provenientes de un picoharp #### (poner modelo).
El programa toma datos de eventos por píxel, separa los fotones correspondientes a la ida y vuelta, y grafica ambas
imágenes. 
@author: Luis1
"""
import pandas as pd
import read_PTU_pixels_2 as rd
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import sys
from PIL import Image
from scipy.signal import find_peaks
plt.style.use(r"C:\Users\Luis1\Downloads\gula_style.mplstyle")
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"

path = "C:\\Users\\Luis1\\Downloads\\"

sys.path.append(path)
import matplotvanda as vd

# ----- Funciones para graficar ida y vuelta -----

def graficar_ida(x,y,imagen):
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')   # origin='lower' para que y vaya de abajo hacia arriba

    ax.set_title("(ida) ", fontsize=12, fontweight='bold')
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')    
def graficar_vuelta(x,y,imagen):
    imagen = np.flip(imagen, axis=1)
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')   # origin='lower' para que y vaya de abajo hacia arriba
    
    ax.set_title("(Vuelta) ", fontsize=12, fontweight='bold')

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')
    plt.show()

# ----- Separa en ida y vuelta, cortando pixeles de mas del final -----

def separar_ida_vuelta(pixeles_ida,pixeles_vuelta, shape,n_pix_objetivo):
    cuentas_ida = np.array([len(x) for x in pixeles_ida])  
    cuentas_vuelta = np.array([len(x) for x in pixeles_vuelta])  
    # recortar o rellenar como antes
    if len(cuentas_ida) < n_pix_objetivo:
        cuentas_ida = np.pad(cuentas_ida, (0, n_pix_objetivo - len(cuentas_ida)), mode='constant', constant_values=0)
    elif len(cuentas_ida) > n_pix_objetivo:
        cuentas_ida = cuentas_ida[:n_pix_objetivo]
    if len(cuentas_vuelta) < n_pix_objetivo:
         cuentas_vuelta = np.pad(cuentas_vuelta, (0, n_pix_objetivo - len(cuentas_vuelta)), mode='constant', constant_values=0)
    elif len(cuentas_vuelta) > n_pix_objetivo:
        cuentas_vuelta = cuentas_vuelta[:n_pix_objetivo]
    imagen_ida = cuentas_ida.reshape(shape)
    imagen_vuelta = cuentas_vuelta.reshape(shape)
    return imagen_ida, imagen_vuelta



def imagen_ida_vuelta(file, n_pix, tamano_um,pixeles_ida_centro ):
    archivo = path + file + ".ptu"
    shape = (n_pix,n_pix)
    n_pix_objetivo = shape[0]*shape[1]
    x = np.linspace(0, tamano_um, shape[1])  # horizontal (cols)
    y = np.linspace(0, tamano_um, shape[0])  # vertical (rows)
    with open(archivo, 'rb') as fd:
        numRecords, _, _ = rd.readHeaders(fd)
        _, _, pixeles = rd.readPT3_fast_pixels(fd, numRecords)
        # Filtramos los píxeles (ignoramos bordes)
        _, pixeles_ida, pixeles_vuelta = rd.filtrar_pixels(pixeles[pixeles_ida_centro:len(pixeles)], n_pix, 4, 1)
        imagen_ida, _ = separar_ida_vuelta(pixeles_ida,pixeles_vuelta,shape,n_pix_objetivo)
        _, imagen_vuelta = separar_ida_vuelta(pixeles_ida,pixeles_vuelta,shape,n_pix_objetivo)
        return x, y, imagen_ida, imagen_vuelta  

# ----- Guardamos .tiff para ImageJ -----
def guardar_imagen_tiff(file, imagen_ida, imagen_vuelta):
    tiff.imwrite( path + file + "_ida.tif", imagen_ida.astype(np.float32))
    tiff.imwrite( path + file + "_vuelta.tif", np.flip(imagen_vuelta, axis = 1).astype(np.float32))
    

# -----Análisis del perfil de intensidad para ver separación entre ida y vuelta -----
def delta_ida_vuelta(file_name, px_size, dwell_time):
    datos = pd.read_csv(path + file_name + ".csv")
    x = datos["Distance_(_)"]*px_size
    y = datos["Gray_Value"]
    peaks, properties = find_peaks(y, prominence=0.05*np.max(y), distance=5)
    if len(peaks) >= 2:
        # ordenar por altura de pico
        top2_idx = np.argsort(y[peaks])[-2:]
        peaks = peaks[top2_idx]
        peaks = np.sort(peaks)  # ordenar de izquierda a derecha
    
        # distancia entre picos (en µm)
        separation = np.abs(x[peaks[1]] - x[peaks[0]])
    else:
        separation = np.nan
        print(" No se detectaron dos picos claros.")
    
    # --- Graficar ---
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(x, y, color='mediumblue', lw=1.5)
    ax.plot(x[peaks], y[peaks], "ro")
    
    if not np.isnan(separation):
        plt.axvline(x[peaks[0]], color='r', ls='--', alpha=0.6)
        plt.axvline(x[peaks[1]], color='r', ls='--', alpha=0.6)
        plt.text(np.mean(x[peaks]), np.max(y)*0.9, f"Δx = {separation:.2f} µm",
                 ha='center', color='r', fontsize=11, fontweight='bold')
       
    
        # Trazar la línea entre los dos puntos
        plt.plot([x[peaks[0]], x[peaks[1]]], [y[peaks[1]], y[peaks[1]]], color = "red")
     
    plt.text(0, np.max(y),f"$V_s$: {px_size/dwell_time:.3f} µm/µs",
              ha='left', color='black', fontsize=12)
    ax.set_xlabel("Distancia (µm)")
    ax.set_ylabel("Intensidad")
    ax.legend()
    vd.gula_grid(ax)
    plt.show()
    print(f"Separación entre emisores: {separation:.3f} µm")




if __name__ == "__main__":
    file = "5x5um-50px-005ms"
    number_of_pixels = 50
    px_size  = 1/10
    image_size_um = 5
    pixeles_ida_al_cero = 21
    dwell_time = 50
    x, y, imagen_ida, imagen_vuelta = imagen_ida_vuelta(file, number_of_pixels,
    image_size_um, pixeles_ida_al_cero)
    graficar_ida(x,y,imagen_ida)
    graficar_vuelta(x,y,imagen_vuelta)
    # guardar_imagen_tiff(file, imagen_ida, imagen_vuelta)
    delta_ida_vuelta(file, px_size, dwell_time)


