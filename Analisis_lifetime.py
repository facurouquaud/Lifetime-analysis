# -*- coding: utf-8 -*-
"""
Archivo para estudiar las imágenes de lifetime, con datos provenientes de un picoharp #### (poner modelo).
El programa toma datos de eventos por píxel, separa los fotones correspondientes a la ida y vuelta, y grafica ambas
imágenes. 
@author: Luis1
"""
import read_PTU_pixels_2 as rd
import matplotlib as plt
import numpy as np


# ----- Cargamos los archivos y los leemos (están en formato PTU) -----

archivo_1 = "C:\\Users\\Luis1\\Downloads\\Fotobeadsarribaizq.ptu"
archivo_2 = "C:\\Users\\Luis1\\Downloads\\centrobead.ptu"

datos = [archivo_1, archivo_2]
pixeles_ida_12 = []
pixeles_vuelta_12 = []

for i in range(2):
    with open(datos[i], 'rb') as fd:
        numRecords, _, _ = rd.readHeaders(fd)
        _, _, pixeles = rd.readPT3_fast_pixels(fd, numRecords)
        
        # Filtramos los píxeles (ignoramos bordes)
        _, pixeles_ida, pixeles_vuelta = rd.filtrar_pixels(pixeles[36:len(pixeles) - 36], 500, 4, 1)
        
        # Guardamos los resultados de cada archivo
        pixeles_ida_12.append(pixeles_ida)
        pixeles_vuelta_12.append(pixeles_vuelta)


# ----- parámetros de la imágen -----
shape = (500, 500)
tamano_um = 10.0
n_pix_objetivo = np.prod(shape)
x = np.linspace(0, tamano_um, shape[1])  # horizontal (cols)
y = np.linspace(0, tamano_um, shape[0])  # vertical (rows)



# ----- Separamos en ida y vuelta -----
imagen_ida = []
imagen_vuelta = []
for i in range(2):
    cuentas_ida = np.array([len(x) for x in pixeles_ida_12[i]])  
    cuentas_vuelta = np.array([len(x) for x in pixeles_vuelta_12[i]])  
    # recortar o rellenar como antes
    if len(cuentas_ida) < n_pix_objetivo:
        cuentas_ida = np.pad(cuentas_ida, (0, n_pix_objetivo - len(cuentas_ida)), mode='constant', constant_values=0)
    elif len(cuentas_ida) > n_pix_objetivo:
        cuentas_ida = cuentas_ida[:n_pix_objetivo]
    if len(cuentas_vuelta) < n_pix_objetivo:
         cuentas_vuelta = np.pad(cuentas_vuelta, (0, n_pix_objetivo - len(cuentas_vuelta)), mode='constant', constant_values=0)
    elif len(cuentas_vuelta) > n_pix_objetivo:
        cuentas_vuelta = cuentas_vuelta[:n_pix_objetivo]
    imagen_ida.append(cuentas_ida.reshape(shape))
    imagen_vuelta.append(cuentas_vuelta.reshape(shape))
    

def graficar_ida(x,y,imagen):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')   # origin='lower' para que y vaya de abajo hacia arriba

    ax.set_title("Reconstrucción (vuelta) — 10 µm × 10 µm", fontsize=12, fontweight='bold')
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

    ax.set_title("Reconstrucción (vuelta) — 10 µm × 10 µm", fontsize=12, fontweight='bold')
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    

#graficamos la ida y vuelta para nuestras dos imagenes
for i in range(2):
    graficar_ida(x,y,imagen_ida[i])
    graficar_vuelta(x,y,imagen_vuelta[i])


