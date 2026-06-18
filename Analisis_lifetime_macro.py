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

def imagen_ida_vuelta_desde_line_markers(
    path, file,
    n_pix_img,   # píxeles útiles por línea
    n_pix_acc,   # píxeles de acel/freno por rampa (4)
    tamano_um,
    dwell_ns,    # dwell por píxel útil, en ns
    N_lineas=None,   # líneas por frame; si None → una sola imagen con todas las líneas
    lifetime_ns=None,
    bin_width_ns=None
):
    """
    Reconstruye stacks ida/vuelta usando:
      - markers al INICIO y FIN de la IDA de cada línea (dos por línea),
      - PERO solo usa los markers de INICIO como referencia de línea,
      - macrotime + dwell_ns,
      - patrón completo de línea con accel/freno e ida/vuelta.

    Patrón por línea (completo):
      puntos_por_linea = 2*n_pix_img + 4*n_pix_acc

      [0 .. n_pix_acc-1]                               acel ida
      [n_pix_acc .. n_pix_acc+n_pix_img-1]             ida útil
      [n_pix_acc+n_pix_img .. 2*n_pix_acc+n_pix_img-1] freno ida
      [2*n_pix_acc+n_pix_img .. 3*n_pix_acc+n_pix_img-1] acel vuelta
      [3*n_pix_acc+n_pix_img .. 3*n_pix_acc+2*n_pix_img-1] vuelta útil
      [3*n_pix_acc+2*n_pix_img .. 4*n_pix_acc+2*n_pix_img-1] freno vuelta
    """

    archivo = f"{path}\\{file}.ptu"

    # ---------- 1) Leer headers ----------
    with open(archivo, 'rb') as fd:
        numRecords, globRes, timeRes = rd.readHeaders(fd)

    T_sync_ns = globRes * 1e9                      # ns por tick truensync
    dwell_sync = int(round(dwell_ns / T_sync_ns))  # ticks por píxel útil

    # ---------- 2) Leer eventos ----------
    T3WRAPAROUND = 65536
    ofl = 0

    truensync_ph = []   # macrotime fotones
    dtime_ph     = []   # microtime fotones
    truensync_mk = []   # macrotime markers

    with open(archivo, 'rb') as fd:
        rd.readHeaders(fd)

        for _ in range(numRecords):
            raw = fd.read(4)
            if not raw:
                break
            recordData = struct.unpack('<I', raw)[0]

            channel = recordData >> 28
            dtime   = (recordData >> 16) & 0xFFF
            nsync   = recordData & 0xFFFF

            if channel == 15:
                # OVERFLOW / MARKER
                if dtime == 0:
                    ofl += T3WRAPAROUND
                else:
                    truensync_mk.append(ofl + nsync)
            elif channel in (1, 2):
                truensync_ph.append(ofl + nsync)
                dtime_ph.append(dtime)
            else:
                pass

    truensync_ph = np.array(truensync_ph, dtype=np.int64)
    dtime_ph     = np.array(dtime_ph,     dtype=np.int32)
    truensync_mk = np.array(truensync_mk, dtype=np.int64)
    print(len(truensync_mk))



    if truensync_mk.size < 2:
        raise RuntimeError("Se esperaban al menos dos markers.")

    # ---------- 3) Markers → inicios de línea ----------
    # Ahora hay 2 markers por línea: inicio_ida, fin_ida.
    # Tomamos solo los de INICIO (asumiendo orden: inicio, fin, inicio, fin, ...)
    truensync_mk = np.sort(truensync_mk)
    if truensync_mk.size > 1:
        # quitar duplicados consecutivos si los hubiera
        mask_uniq = np.concatenate(([True], np.diff(truensync_mk) != 0))
        truensync_mk = truensync_mk[mask_uniq]

    # Asegurar número PAR de markers (pares inicio/fin)
    if truensync_mk.size % 2 != 0:
        truensync_mk = truensync_mk[:-1]

    # Inicios de línea = primeros de cada par
    line_starts = truensync_mk[0::2]
    n_lineas_totales = line_starts.size

    if n_lineas_totales == 0:
        raise RuntimeError("No se detectaron líneas a partir de los markers.")

    # Duración completa de una línea (ida+vuelta) en ticks
    puntos_por_linea = 2 * n_pix_img + 4 * n_pix_acc
    dur_line_sync = puntos_por_linea * dwell_sync

    if N_lineas is None:
        N_lineas = n_lineas_totales
    n_frames = n_lineas_totales // N_lineas
    if n_frames == 0:
        raise ValueError("No hay suficientes líneas para formar un frame completo.")

    # ---------- 4) Filtro lifetime ----------
    if lifetime_ns is not None:
        if bin_width_ns is None:
            raise ValueError("bin_width_ns debe especificarse para lifetime_ns.")
        t_ns = dtime_ph * bin_width_ns
        t_min, t_max = lifetime_ns
        mask_life_global = (t_ns >= t_min) & (t_ns <= t_max)
    else:
        mask_life_global = np.ones_like(dtime_ph, dtype=bool)

    # ---------- 5) Stacks de salida ----------
    ida_stack    = np.zeros((n_frames, N_lineas, n_pix_img), dtype=np.int32)
    vuelta_stack = np.zeros((n_frames, N_lineas, n_pix_img), dtype=np.int32)

    # ---------- 6) Procesar línea por línea ----------
    for line_global in range(n_lineas_totales):
        t0_line = line_starts[line_global]
        t1_line = t0_line + dur_line_sync  # FIN de línea completa (ida+vuelta)

        # Fotones en esta línea completa
        mask_line_time = (truensync_ph >= t0_line) & (truensync_ph < t1_line)
        if not np.any(mask_line_time):
            continue

        mask_line = mask_line_time & mask_life_global
        if not np.any(mask_line):
            continue

        t_line = truensync_ph[mask_line]
        mt_rel_line = t_line - t0_line

        # Índice de "punto" dentro de la línea
        idx_pix_line = (mt_rel_line // dwell_sync).astype(np.int64)
        if idx_pix_line.size == 0:
            continue

        punto_en_linea = idx_pix_line  # 0..puntos_por_linea-1 (aprox)

        # Máscaras de ida/vuelta útiles
        ida_mask = (punto_en_linea >= n_pix_acc) & \
                   (punto_en_linea <  n_pix_acc + n_pix_img)

        vuelta_mask = (punto_en_linea >= 3 * n_pix_acc + n_pix_img) & \
                      (punto_en_linea <  3 * n_pix_acc + 2 * n_pix_img)

        if not (np.any(ida_mask) or np.any(vuelta_mask)):
            continue

        col_ida    = punto_en_linea[ida_mask]    - n_pix_acc
        col_vuelta = punto_en_linea[vuelta_mask] - (3 * n_pix_acc + n_pix_img)

        frame_idx = line_global // N_lineas
        y_idx     = line_global %  N_lineas
        if frame_idx >= n_frames:
            break

        for c in col_ida:
            if 0 <= c < n_pix_img:
                ida_stack[frame_idx, y_idx, c] += 1

        for c in col_vuelta:
            if 0 <= c < n_pix_img:
                vuelta_stack[frame_idx, y_idx, c] += 1

    # ---------- 7) Coordenadas espaciales ----------
    x = np.linspace(0, tamano_um, n_pix_img)
    y = np.linspace(0, tamano_um, N_lineas)

    return x, y, ida_stack, vuelta_stack, n_frames





if __name__ == "__main__":
    print("Version 1.0.00")

    path = r"C:\Users\Luis1\Downloads"
    file = "6"

    n_pix_img = 120
    n_pix_acc = 4
    N_lineas  = 120
    tamano_um = 6

    dwell_ns = 600.0 * 1e3   # ej. 50 µs = 50e3 ns

    bin_width_ns = 0.032
    lifetime_ns  = None      # o (5,8) si querés filtrar lifetime

    x, y, ida_stack, vuelta_stack, n_frames = imagen_ida_vuelta_desde_line_markers(
        path, file,
        n_pix_img=n_pix_img,
        n_pix_acc=n_pix_acc,
        tamano_um=tamano_um,
        dwell_ns=dwell_ns,
        N_lineas=N_lineas,
        lifetime_ns=lifetime_ns,
        bin_width_ns=bin_width_ns
    )

    print(f"{n_frames} frames completos (líneas con markers inicio/fin)")

    for f in range(min(3, n_frames)):
        graficar_ida(x, y, ida_stack[f], titulo=f"Ida frame {f}")
        graficar_vuelta(x, y, vuelta_stack[f], titulo=f"Vuelta frame {f}")


 





