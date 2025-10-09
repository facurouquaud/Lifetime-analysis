# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 20:40:29 2025

@author: Lenovo
"""

import time
import struct
import numpy as np
import matplotlib.pyplot as plt
# Tipos de datos de la cabecera PTU
tyEmpty8 = struct.unpack('>i', bytes.fromhex('FFFF0008'))[0]
tyBool8 = struct.unpack('>i', bytes.fromhex('00000008'))[0]
tyInt8 = struct.unpack('>i', bytes.fromhex('10000008'))[0]
tyBitSet64 = struct.unpack('>i', bytes.fromhex('11000008'))[0]
tyColor8 = struct.unpack('>i', bytes.fromhex('12000008'))[0]
tyFloat8 = struct.unpack('>i', bytes.fromhex('20000008'))[0]
tyTDateTime = struct.unpack('>i', bytes.fromhex('21000008'))[0]
tyFloat8Array = struct.unpack('>i', bytes.fromhex('2001FFFF'))[0]
tyAnsiString = struct.unpack('>i', bytes.fromhex('4001FFFF'))[0]
tyWideString = struct.unpack('>i', bytes.fromhex('4002FFFF'))[0]
tyBinaryBlob = struct.unpack('>i', bytes.fromhex('FFFFFFFF'))[0]


def readHeaders(inputfile):
    magic = inputfile.read(8).decode('utf-8').strip('\x00')
    if magic != 'PQTTTR':
        raise ValueError('ERROR: Magic invalid, this is not a PTU file.')
    version = inputfile.read(8).decode('utf-8').strip('\x00')
    print('Version', version)
    tagDataList = []
    while True:
        tagIdent = inputfile.read(32).decode('utf-8').strip('\x00')
        tagIdx = struct.unpack('<i', inputfile.read(4))[0]
        tagTyp = struct.unpack('<i', inputfile.read(4))[0]
        if tagIdx > -1:
            evalName = tagIdent + '(' + str(tagIdx) + ')'
        else:
            evalName = tagIdent
        if tagTyp == tyEmpty8:
            inputfile.read(8)
            tagDataList.append((evalName, '<empty Tag>'))
        elif tagTyp == tyBool8:
            tagInt = struct.unpack('<q', inputfile.read(8))[0]
            tagDataList.append((evalName, bool(tagInt)))
        elif tagTyp == tyInt8 or tagTyp == tyBitSet64 or tagTyp == tyColor8:
            tagInt = struct.unpack('<q', inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyFloat8 or tagTyp == tyTDateTime:
            tagFloat = struct.unpack('<d', inputfile.read(8))[0]
            tagDataList.append((evalName, tagFloat))
        elif tagTyp == tyAnsiString:
            tagInt = struct.unpack('<q', inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode('utf-8').strip('\x00')
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyWideString:
            tagInt = struct.unpack('<q', inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode('utf-16le', errors='ignore').strip('\x00')
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyBinaryBlob:
            tagInt = struct.unpack('<q', inputfile.read(8))[0]
            inputfile.read(tagInt)  # se salta los datos binarios
            tagDataList.append((evalName, f'<binary blob of {tagInt} bytes>'))
        else:
            raise ValueError(f'Unknown tag type: {tagTyp}')
        if tagIdent == 'Header_End':
            break
    tagDict = dict(tagDataList)
    numRecords = tagDict['TTResult_NumberOfRecords']
    globRes = tagDict['MeasDesc_GlobalResolution']
    timeRes = tagDict['MeasDesc_Resolution']
    return numRecords, globRes, timeRes


def readPT3_fast_pixels(inputfile, numRecords):
    T3WRAPAROUND = 65536
    oflcorrection = 0
    dlen = 0

    dtime_array = np.zeros(numRecords)
    truensync_array = np.zeros(numRecords)
    pixeles = []
    pixel_actual = []
    marker_events = []

    for recNum in range(numRecords):
        try:
            recordData = struct.unpack('<I', inputfile.read(4))[0]
        except:
            print(f'Archivo terminado antes de lo esperado, evento {recNum}/{numRecords}')
            break

        channel = recordData >> 28
        dtime = (recordData >> 16) & 0xFFF
        nsync = recordData & 0xFFFF

        if channel == 15:
            if dtime == 0:
                oflcorrection += T3WRAPAROUND
            elif dtime == 2:
                # Marker 1: nuevo píxel
                pixeles.append(np.array(pixel_actual))  # se guarda aunque esté vacío
                pixel_actual = []
        elif 0 <= channel <= 4:
            truensync = oflcorrection + nsync
            dtime_array[dlen] = dtime
            truensync_array[dlen] = truensync
            dlen += 1
            pixel_actual.append((dtime, truensync))
        else:
            print(f'Canal ilegal #{channel} en registro {recNum}')

    # Guarda último píxel si tiene datos 
    if pixel_actual:
        pixeles.append(np.array(pixel_actual))

    return dtime_array[:dlen], truensync_array[:dlen], pixeles

if __name__ == "__main__":   
    archivo = "C:\\Users\\Luis1\\Downloads\\centrobead.ptu"
    # archivo = "C:\\Users\\Lenovo\\Downloads\\pruebaDAQ100p.ptu"
    with open(archivo, 'rb') as fd:
        numRecords, _, _ = readHeaders(fd)
        dtime, truesync, pixeles = readPT3_fast_pixels(fd, numRecords)
    
    print(f"Total de pixeles detectados: {len(pixeles)}")
    
def filtrar_pixels(pixels, n_pix, n_pix_acc, N):
    total_pixeles = len(pixels)
    puntos_por_linea = 2 * n_pix + 4 * n_pix_acc
    total_por_escaneo = N * puntos_por_linea

    indices = np.arange(total_pixeles)
    local_idx = indices % total_por_escaneo
    punto_en_linea = local_idx % puntos_por_linea

    # # Máscaras para tramos de velocidad constante (ida o vuelta)
    mascara_ida = (punto_en_linea >= n_pix_acc) & (punto_en_linea < n_pix_acc + n_pix)
    mascara_vuelta = (punto_en_linea >= 3 * n_pix_acc + n_pix) & (punto_en_linea < 3 * n_pix_acc + 2 * n_pix)
    mascara_valida = mascara_ida | mascara_vuelta

    pixeles_validos = [px for i, px in enumerate(pixels) if mascara_valida[i]]
    pixeles_ida = [px for i, px in enumerate(pixels) if mascara_ida[i]]
    pixeles_vuelta = [px for i, px in enumerate(pixels) if mascara_vuelta[i]]

    return pixeles_validos, pixeles_ida, pixeles_vuelta
#%%
archivo_1 = "C:\\Users\\Luis1\\Downloads\\Fotobeadsarribaizq.ptu"
archivo_2 = "C:\\Users\\Luis1\\Downloads\\centrobead.ptu"

datos = [archivo_1, archivo_2]
pixeles_ida_12 = []
pixeles_vuelta_12 = []

for i in range(2):
    with open(datos[i], 'rb') as fd:
        numRecords, _, _ = readHeaders(fd)
        _, _, pixeles = readPT3_fast_pixels(fd, numRecords)
        
        # Filtramos los píxeles (ignoramos bordes)
        _, pixeles_ida, pixeles_vuelta = filtrar_pixels(pixeles[36:len(pixeles) - 36], 500, 4, 1)
        
        # Guardamos los resultados de cada archivo
        pixeles_ida_12.append(pixeles_ida)
        pixeles_vuelta_12.append(pixeles_vuelta)

#%%

# ----- parámetros -----
shape = (500, 500)
tamano_um = 10.0
n_pix_objetivo = np.prod(shape)
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
    

#imagenes ida
x = np.linspace(0, tamano_um, shape[1])  # horizontal (cols)
y = np.linspace(0, tamano_um, shape[0])  # vertical (rows)
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
    
# for i in range(2):
#     imagen = np.flip(imagen, axis=1)     

# fig, ax = plt.subplots(figsize=(6, 6))
# im = ax.imshow(imagen, cmap='inferno',
#                extent=[x.min(), x.max(), y.min(), y.max()],
#                origin='lower')   # origin='lower' para que y vaya de abajo hacia arriba

# ax.set_title("Reconstrucción (vuelta) — 10 µm × 10 µm", fontsize=12, fontweight='bold')
# ax.set_xlabel("x [µm]")
# ax.set_ylabel("y [µm]")
# fig.colorbar(im, ax=ax, label="Número de fotones")
# ax.set_aspect('equal', adjustable='box')
# plt.tight_layout()
# plt.show()
for i in range(2):
    graficar_ida(x,y,imagen_ida[i])
    graficar_vuelta(x,y,imagen_vuelta[i])


#%%
# imagen = np.fliplr(imagen)
