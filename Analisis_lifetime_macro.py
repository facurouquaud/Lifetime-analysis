# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:35:01 2026

@author: Luis1
"""
import read_PTU_pixels_2 as rd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector  
from scipy.optimize import curve_fit

def graficar_ida(x, y, imagen, titulo="Ida"):
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')
    ax.set_xlabel("x [µm]", fontsize=14)
    ax.set_ylabel("y [µm]", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Número de fotones", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(titulo)
    plt.show()


def graficar_vuelta(x, y, imagen, titulo="Vuelta"):
    imagen = np.flip(imagen, axis=1)
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')
    ax.set_xlabel("x [µm]", fontsize=14)
    ax.set_ylabel("y [µm]", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Número de fotones", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(titulo)
    plt.show()
import struct

import struct

def encontrar_ventana_markers(archivo_ptu):
    """
    Devuelve (t0, t1) = (truensync del PRIMER marker real, truensync del ÚLTIMO marker real)
    suponiendo:
      - channel == 15, dtime == 0 → overflow
      - channel == 15, dtime != 0 → marker de sincronización.
    """
    T3WRAPAROUND = 65536
    ofl = 0

    t_markers = []

    with open(archivo_ptu, 'rb') as fd:
        numRecords, _, _ = rd.readHeaders(fd)

        for _ in range(numRecords):
            try:
                recordData = struct.unpack('<I', fd.read(4))[0]
            except:
                break

            channel = recordData >> 28
            dtime   = (recordData >> 16) & 0xFFF
            nsync   = recordData & 0xFFFF

            if channel == 15:
                if dtime == 0:
                    ofl += T3WRAPAROUND
                else:
                    truensync = ofl + nsync
                    t_markers.append(truensync)

    if len(t_markers) < 2:
        raise RuntimeError("Se esperaban al menos dos marcadores (inicio y fin de escaneo).")

    t_markers = np.array(t_markers, dtype=np.int64)
    t0 = t_markers[0]
    t1 = t_markers[-1]
    return t0, t1


    raise RuntimeError("No se encontró ningún marker de sincronización en el archivo.")
def imagen_ida_vuelta_macrotime_entre_markers(path, file,
                                              n_pix_img,   # píxeles útiles por línea
                                              n_pix_acc,   # píxeles de acel/freno (4)
                                              N_lineas,    # líneas por imagen
                                              tamano_um,
                                              dwell_sync,  # dwell en ticks truensync
                                              lifetime_ns=None,
                                              bin_width_ns=None,
                                              offset_pix=0,        # corrimiento en píxeles crudos
                                              n_pix_gap_frame=0    # píxeles crudos entre frames
                                              ):
    """
    Reconstruye stacks ida/vuelta usando:
      - ventana temporal acotada por markers de inicio/fin (t0, t1),
      - macrotime + dwell_sync,
      - patrón de línea con accel/freno e ida/vuelta,
      - y filtrando el retrazo vertical entre frames (n_pix_gap_frame).
    """

    archivo = f"{path}\\{file}.ptu"

    # 1) Ventana [t0, t1] por markers
    t0, t1 = encontrar_ventana_markers(archivo)

    # 2) Leer fotones
    with open(archivo, 'rb') as fd:
        numRecords, _, _ = rd.readHeaders(fd)
        dtime_all, truensync_all, _ = rd.readPT3_fast_pixels(fd, numRecords)

    # 3) Filtro lifetime (opcional)
    if lifetime_ns is not None:
        if bin_width_ns is None:
            raise ValueError("bin_width_ns debe especificarse para lifetime_ns.")
        t_ns = dtime_all * bin_width_ns
        t_min, t_max = lifetime_ns
        mask_life = (t_ns >= t_min) & (t_ns <= t_max)
    else:
        mask_life = np.ones_like(dtime_all, dtype=bool)

    # 4) Solo fotones en [t0, t1]
    mask_win = (truensync_all >= t0) & (truensync_all <= t1)
    mask = mask_life & mask_win
    if not np.any(mask):
        raise ValueError("No hay fotones dentro de [t0, t1].")

    truensync_use = truensync_all[mask]

    # 5) Macrotime relativo y píxel crudo global
    mt_rel = truensync_use - t0
    idx_pix_total = (mt_rel // dwell_sync).astype(np.int64)

    # Aplicar offset en píxeles crudos (tiempo de respuesta, etc.)
    idx_pix_total = idx_pix_total - offset_pix
    mask_pos = idx_pix_total >= 0
    idx_pix_total = idx_pix_total[mask_pos]

    if idx_pix_total.size == 0:
        raise ValueError("Todos los fotones quedaron antes de offset_pix.")

    total_pix = idx_pix_total.max() + 1  # 0..total_pix-1

    # 6) Geometría de línea y frame
    puntos_por_linea   = 2 * n_pix_img + 4 * n_pix_acc
    total_por_escaneo  = N_lineas * puntos_por_linea   # píxeles crudos útiles por frame
    frame_block        = total_por_escaneo + n_pix_gap_frame  # útil + retrazo vertical

    # Número de frames completos
    n_frames = total_pix // frame_block
    if n_frames == 0:
        raise ValueError("No hay suficientes datos para formar ni un frame completo.")

    max_idx_util = n_frames * frame_block
    mask_block = idx_pix_total < max_idx_util
    idx = idx_pix_total[mask_block]

    # 7) Índices locales dentro del bloque de cada frame
    frame_idx    = idx // frame_block
    dentro_block = idx %  frame_block

    # Solo la parte útil del frame (antes del retrazo vertical)
    mask_in_frame = dentro_block < total_por_escaneo
    frame_idx     = frame_idx[mask_in_frame]
    local_idx     = dentro_block[mask_in_frame]  # 0..total_por_escaneo-1

    # Línea y posición dentro de línea (MISMA lógica que filtrar_pixels)
    line_idx       = local_idx // puntos_por_linea
    punto_en_linea = local_idx %  puntos_por_linea

    # Limitar a líneas válidas
    mask_line = (line_idx >= 0) & (line_idx < N_lineas)
    frame_idx      = frame_idx[mask_line]
    line_idx       = line_idx[mask_line]
    punto_en_linea = punto_en_linea[mask_line]

    if frame_idx.size == 0:
        raise ValueError("No hay fotones en líneas válidas dentro de la ventana.")

    # 8) Máscaras de ida/vuelta útiles (descartando accel/freno)
    ida_mask = (punto_en_linea >= n_pix_acc) & (punto_en_linea < n_pix_acc + n_pix_img)
    vuelta_mask = (punto_en_linea >= 3 * n_pix_acc + n_pix_img) & \
                  (punto_en_linea <  3 * n_pix_acc + 2 * n_pix_img)

    col_ida    = punto_en_linea[ida_mask]    - n_pix_acc
    col_vuelta = punto_en_linea[vuelta_mask] - (3 * n_pix_acc + n_pix_img)

    row_ida    = line_idx[ida_mask]
    row_vuelta = line_idx[vuelta_mask]

    frame_ida    = frame_idx[ida_mask]
    frame_vuelta = frame_idx[vuelta_mask]

    if frame_ida.size == 0 and frame_vuelta.size == 0:
        raise ValueError("No hay fotones en tramos de ida/vuelta útiles.")

    # 9) Acumular en stacks
    ida_stack    = np.zeros((n_frames, N_lineas, n_pix_img), dtype=np.int32)
    vuelta_stack = np.zeros((n_frames, N_lineas, n_pix_img), dtype=np.int32)

    for f, r, c in zip(frame_ida, row_ida, col_ida):
        if 0 <= f < n_frames and 0 <= r < N_lineas and 0 <= c < n_pix_img:
            ida_stack[f, r, c] += 1

    for f, r, c in zip(frame_vuelta, row_vuelta, col_vuelta):
        if 0 <= f < n_frames and 0 <= r < N_lineas and 0 <= c < n_pix_img:
            vuelta_stack[f, r, c] += 1

    # 10) Coordenadas espaciales
    x = np.linspace(0, tamano_um, n_pix_img)
    y = np.linspace(0, tamano_um, N_lineas)

    return x, y, ida_stack, vuelta_stack, n_frames




if __name__ == "__main__":
    path = r"C:\Users\Luis1\Downloads"
    file = "un_frame"

    n_pix_img = 200
    n_pix_acc = 4
    N_lineas  = 200
    tamano_um = 10

    archivo = f"{path}\\{file}.ptu"
    with open(archivo, 'rb') as fd:
        numRecords, globRes, timeRes = rd.readHeaders(fd)

    dwell_ns  = 80.0*1E3
    T_sync_ns = globRes * 1e9
    dwell_sync = int(round(dwell_ns / T_sync_ns))

    bin_width_ns = 0.032
    lifetime_ns  = None

    offset_pix      = 70     # podés tunearlo
    n_pix_gap_frame = 0   # retrazo vertical entre frames (en píxeles crudos)

    x, y, ida_stack, vuelta_stack, n_frames = imagen_ida_vuelta_macrotime_entre_markers(
        path=path,
        file=file,
        n_pix_img=n_pix_img,
        n_pix_acc=n_pix_acc,
        N_lineas=N_lineas,
        tamano_um=tamano_um,
        dwell_sync=dwell_sync,
        lifetime_ns=lifetime_ns,
        bin_width_ns=bin_width_ns,
        offset_pix=offset_pix,
        n_pix_gap_frame=n_pix_gap_frame
    )

    print(f"{n_frames} frames completos (entre markers, con retrazo vertical)")

    # Visualizar algunos frames
    for f in range(min(3, n_frames)):
        graficar_ida(x, y, ida_stack[f], titulo=f"Ida frame {f}")
        graficar_vuelta(x, y, vuelta_stack[f], titulo=f"Vuelta frame {f}")


 





