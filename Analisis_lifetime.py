# -*- coding: utf-8 -*-
"""
Archivo para estudiar las imágenes de lifetime, con datos provenientes de un picoharp #### (poner modelo).
El programa toma datos de eventos por píxel, separa los fotones correspondientes a la ida y vuelta, y grafica ambas
imágenes. 
@author: Luis1
"""
import read_PTU_pixels_2 as rd
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import profile_line
import tifffile as tiff





# ----- Separamos en ida y vuelta -----

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

   
    

def graficar_ida(x,y,imagen):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')   # origin='lower' para que y vaya de abajo hacia arriba

    ax.set_title("(ida) ", fontsize=12, fontweight='bold')
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    
def graficar_vuelta(x,y,imagen):
    imagen = np.flip(imagen, axis=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')   # origin='lower' para que y vaya de abajo hacia arriba
    
    ax.set_title("(Vuelta) ", fontsize=12, fontweight='bold')

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def imagen_ida_vuelta(file, n_pix, tamano_um,pixeles_ida_centro ):
    archivo = "C:\\Users\\Luis1\\Downloads\\" + file + ".ptu"
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
def guardar_imagen_tiff(file, imagen_ida, imagen_vuelta):
    tiff.imwrite("C:\\Users\\Luis1\\Downloads\\" + file + "_ida.tif", imagen_ida.astype(np.float32))
    tiff.imwrite("C:\\Users\\Luis1\\Downloads\\" + file + "_vuelta.tif", np.flip(imagen_vuelta, axis = 1).astype(np.float32))
#%%
# ----- Cargamos los archivos y los leemos (están en formato PTU) -----
# 
# archivo = "C:\\Users\\Luis1\\Downloads\\10x10um-200px-001ms.ptu"
#50x50umm-500px-001ms
file = "Fotobeadsarribaizq"
  
x,y,imagen_ida, imagen_vuelta = imagen_ida_vuelta(file, 500, 10, 16)
graficar_ida(x,y,imagen_ida)
graficar_vuelta(x,y,imagen_vuelta)
guardar_imagen_tiff(file, imagen_ida, imagen_vuelta)




#%%
