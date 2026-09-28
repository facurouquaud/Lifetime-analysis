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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import numpy as np
import tifffile

# Construir colormap: negro -> YlOrBr (usamos los colores desde YlOrBr pero empezando en amarillo/ambar)



import struct

def graficar_ida(x, y, imagen, titulo="Ida"):
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap="viridis",
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')
    ax.set_xlabel("x [µm]", fontsize=20)
    ax.set_ylabel("y [µm]", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Número de fotones", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_aspect('equal', adjustable='box')
   # ax.set_title(titulo)
    plt.show()

def graficar_vuelta(x, y, imagen, titulo="Vuelta"):
    imagen = np.flip(imagen, axis=1)
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='viridis',
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
    path,
    file,
    n_pix_img,
    n_pix_acc,
    tamano_um,
    dwell_ns,
    N_lineas=None,
    lifetime_ns=None,
    bin_width_ns=None,
    N_delay = None
):
    """
    Reconstruye stacks de imágenes de barrido en ida y vuelta.

    Se asume que los markers aparecen en este orden:

        marker[0] -> inicio de línea 0
        marker[1] -> final de la ida de línea 0
        marker[2] -> inicio de línea 1
        marker[3] -> final de la ida de línea 1
        ...

    Por tanto, los inicios de línea son:

        truensync_mk[0::2]

    Patrón temporal de una línea completa:

        puntos_por_linea = 2*n_pix_img + 4*n_pix_acc

        [0, n_pix_acc)
            aceleración de ida

        [n_pix_acc, n_pix_acc+n_pix_img)
            ida útil

        [n_pix_acc+n_pix_img, 2*n_pix_acc+n_pix_img)
            frenado de ida

        [2*n_pix_acc+n_pix_img, 3*n_pix_acc+n_pix_img)
            aceleración de vuelta

        [3*n_pix_acc+n_pix_img,
         3*n_pix_acc+2*n_pix_img)
            vuelta útil

        [3*n_pix_acc+2*n_pix_img,
         4*n_pix_acc+2*n_pix_img)
            frenado de vuelta

    La función conserva una última línea aunque no exista el marker
    posterior. Para esa última línea se estima el final usando la
    separación mediana entre inicios de línea.
    """

    import os
    import struct
    import numpy as np

    # ============================================================
    # 0) Validación de parámetros
    # ============================================================

    if n_pix_img <= 0:
        raise ValueError(
            "n_pix_img debe ser mayor que cero."
        )

    if n_pix_acc < 0:
        raise ValueError(
            "n_pix_acc no puede ser negativo."
        )

    if dwell_ns <= 0:
        raise ValueError(
            "dwell_ns debe ser mayor que cero."
        )

    if N_lineas is not None and N_lineas <= 0:
        raise ValueError(
            "N_lineas debe ser mayor que cero."
        )

    if lifetime_ns is not None:
        if bin_width_ns is None:
            raise ValueError(
                "bin_width_ns debe especificarse cuando se usa "
                "lifetime_ns."
            )

        if len(lifetime_ns) != 2:
            raise ValueError(
                "lifetime_ns debe tener el formato (t_min, t_max)."
            )

    archivo = os.path.join(path, f"{file}.ptu")

    if not os.path.isfile(archivo):
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{archivo}"
        )

    # ============================================================
    # 1) Leer headers
    # ============================================================

    with open(archivo, "rb") as fd:
        numRecords, globRes, timeRes = rd.readHeaders(fd)

    T_sync_ns = globRes * 1e9

    if T_sync_ns <= 0:
        raise ValueError(
            "La resolución global del archivo no es válida."
        )

    # Conversión utilizada por tu código original
    dwell_sync = int(round(dwell_ns / T_sync_ns))

    if dwell_sync <= 0:
        raise ValueError(
            "dwell_sync ha resultado cero. Revisa dwell_ns y globRes."
        )

    # Número teórico de puntos de una línea completa
    puntos_por_linea = (
        2 * n_pix_img +
        4 * n_pix_acc
    )

    dur_line_sync_teorica = (
        puntos_por_linea * dwell_sync
    )

    print()
    print("========== PARÁMETROS ==========")
    print("Archivo:", archivo)
    print("Número de registros:", numRecords)
    print("Resolución macrotime:", T_sync_ns, "ns/tick")
    print("dwell_sync:", dwell_sync, "ticks")
    print("Puntos teóricos por línea:", puntos_por_linea)
    print("Duración teórica por línea:",
          dur_line_sync_teorica, "ticks")
    print("=================================")
    print()

    # ============================================================
    # 2) Leer eventos
    # ============================================================

    T3WRAPAROUND = 65536
    ofl = 0

    truensync_ph = []
    dtime_ph = []
    truensync_mk = []

    n_overflows = 0
    n_markers = 0
    n_photons = 0

    with open(archivo, "rb") as fd:
        rd.readHeaders(fd)

        for _ in range(numRecords):

            raw = fd.read(4)

            if not raw or len(raw) != 4:
                break

            recordData = struct.unpack("<I", raw)[0]

            channel = recordData >> 28
            dtime = (recordData >> 16) & 0xFFF
            nsync = recordData & 0xFFFF

            if channel == 15:

                # Según el formato usado en tu código:
                # dtime=0     -> overflow
                # dtime!=0    -> marker
                if dtime == 0:
                    ofl += T3WRAPAROUND
                    n_overflows += 1

                else:
                    truensync_mk.append(ofl + nsync)
                    n_markers += 1

            elif channel in (1, 2):

                truensync_ph.append(ofl + nsync)
                dtime_ph.append(dtime)
                n_photons += 1

    truensync_ph = np.asarray(
        truensync_ph,
        dtype=np.int64
    )

    dtime_ph = np.asarray(
        dtime_ph,
        dtype=np.int32
    )

    truensync_mk = np.asarray(
        truensync_mk,
        dtype=np.int64
    )

    print("========== EVENTOS LEÍDOS ==========")
    print("Fotones:", n_photons)
    print("Markers:", n_markers)
    print("Overflows:", n_overflows)
    print("=====================================")
    print()

    if truensync_mk.size < 2:
        raise RuntimeError(
            "Se esperaban al menos dos markers."
        )

    # ============================================================
    # 3) Ordenar y eliminar markers duplicados
    # ============================================================

    truensync_mk = np.sort(truensync_mk)

    if truensync_mk.size > 1:

        mask_uniq = np.concatenate(
            (
                np.array([True], dtype=bool),
                np.diff(truensync_mk) != 0
            )
        )

        truensync_mk = truensync_mk[mask_uniq]

    if truensync_mk.size < 2:
        raise RuntimeError(
            "No quedan suficientes markers después de eliminar "
            "duplicados."
        )

    # ============================================================
    # 4) Obtener los inicios de línea
    # ============================================================

    # IMPORTANTE:
    #
    # No hacemos esto:
    #
    # if truensync_mk.size % 2 != 0:
    #     truensync_mk = truensync_mk[:-1]
    #
    # porque si el último marker es el inicio de una última línea,
    # esa línea se perdería.
    #
    # Primero extraemos los inicios de línea. El último inicio se
    # conserva aunque no tenga un marker posterior.

    line_starts = truensync_mk[0::2]

    n_lineas_totales = line_starts.size

    if n_lineas_totales == 0:
        raise RuntimeError(
            "No se detectaron inicios de línea."
        )

    # ============================================================
    # 5) Estimar la duración real entre líneas
    # ============================================================

    if line_starts.size > 1:

        duraciones_linea_observadas = np.diff(line_starts)

        # La mediana es robusta frente a algún intervalo anómalo.
        dur_line_sync_real = int(
            round(
                np.median(duraciones_linea_observadas)
            )
        )

    else:
        dur_line_sync_real = dur_line_sync_teorica

    if dur_line_sync_real <= 0:
        raise RuntimeError(
            "La duración estimada de línea no es válida."
        )

    print("========== MARKERS ==========")
    print("Markers únicos:", len(truensync_mk))
    print("Inicios de línea:", n_lineas_totales)
    print("Duración teórica de línea:",
          dur_line_sync_teorica, "ticks")
    print("Duración real mediana de línea:",
          dur_line_sync_real, "ticks")

    if line_starts.size > 1:
        print(
            "Primeras separaciones entre inicios:",
            np.diff(line_starts)[:10]
        )

        print(
            "Últimas separaciones entre inicios:",
            np.diff(line_starts)[-10:]
        )

    print("==============================")
    print()

    # ============================================================
    # 6) Frames
    # ============================================================

    if N_lineas is None:
        N_lineas = n_lineas_totales

    # Se usa ceil para no perder las líneas restantes.
    #
    # Ejemplo:
    #   50 líneas y N_lineas=40
    #   -> 2 frames: uno de 40 y otro de 10 líneas
    #
    n_frames = int(
        np.ceil(n_lineas_totales / N_lineas)
    )

    if n_frames <= 0:
        raise ValueError(
            "No se pudo construir ningún frame."
        )

    print("========== FRAMES ==========")
    print("Líneas totales:", n_lineas_totales)
    print("Líneas por frame:", N_lineas)
    print("Número de frames:", n_frames)
    print("=============================")
    print()

    # ============================================================
    # 7) Filtro de lifetime
    # ============================================================

    if lifetime_ns is not None:

        t_min, t_max = lifetime_ns

        t_ns = (
            dtime_ph.astype(np.float64) *
            float(bin_width_ns)
        )

        mask_life_global = (
            (t_ns >= t_min) &
            (t_ns <= t_max)
        )

    else:

        mask_life_global = np.ones(
            dtime_ph.shape,
            dtype=bool
        )

    # ============================================================
    # 8) Crear stacks
    # ============================================================

    ida_stack = np.zeros(
        (
            n_frames,
            N_lineas,
            n_pix_img
        ),
        dtype=np.int32
    )

    vuelta_stack = np.zeros(
        (
            n_frames,
            N_lineas,
            n_pix_img
        ),
        dtype=np.int32
    )

    # ============================================================
    # 9) Procesar las líneas
    # ============================================================

    n_lineas_procesadas = 0
    n_lineas_sin_fotones = 0

    for line_global in range(n_lineas_totales):

        # Inicio real de esta línea
        t0_line = int(
            line_starts[line_global]
        )

        # El límite de la línea es el inicio de la siguiente.
        if line_global + 1 < n_lineas_totales:

            t1_line = int(
                line_starts[line_global + 1]
            )

            linea_extrapolada = False

        else:

            # Para la última línea no existe el siguiente inicio.
            # Se utiliza la duración mediana observada.
            t1_line = int(
                t0_line + dur_line_sync_real
            )

            linea_extrapolada = True

        if t1_line <= t0_line:
            print(
                "Advertencia: límite temporal inválido en línea",
                line_global
            )
            continue

        # --------------------------------------------------------
        # Seleccionar fotones dentro de la línea
        # --------------------------------------------------------

        mask_line_time = (
            (truensync_ph >= t0_line) &
            (truensync_ph < t1_line)
        )

        if not np.any(mask_line_time):
            n_lineas_sin_fotones += 1
            n_lineas_procesadas += 1
            continue

        mask_line = (
            mask_line_time &
            mask_life_global
        )

        if not np.any(mask_line):
            n_lineas_sin_fotones += 1
            n_lineas_procesadas += 1
            continue

        # Tiempo relativo al inicio de la línea
        mt_rel_line = (
            truensync_ph[mask_line] - t0_line + N_delay*dwell_sync
        )

        # Índice de punto dentro de la línea
        idx_pix_line = (
            mt_rel_line // dwell_sync
        ).astype(np.int64)

        # --------------------------------------------------------
        # Máscara de ida útil
        # --------------------------------------------------------

        ida_mask = (
            (idx_pix_line >= n_pix_acc) &
            (
                idx_pix_line <
                n_pix_acc + n_pix_img
            )
        )

        # --------------------------------------------------------
        # Máscara de vuelta útil
        # --------------------------------------------------------

        inicio_vuelta_util = (
            3 * n_pix_acc +
            n_pix_img
        )

        fin_vuelta_util = (
            3 * n_pix_acc +
            2 * n_pix_img
        )

        vuelta_mask = (
            (idx_pix_line >= inicio_vuelta_util) &
            (idx_pix_line < fin_vuelta_util)
        )

        # Columnas resultantes
        col_ida = (
            idx_pix_line[ida_mask] -
            n_pix_acc
        )

        col_vuelta = (
            idx_pix_line[vuelta_mask] -
            inicio_vuelta_util
        )

        # Eliminar cualquier columna fuera de rango
        col_ida = col_ida[
            (col_ida >= 0) &
            (col_ida < n_pix_img)
        ]

        col_vuelta = col_vuelta[
            (col_vuelta >= 0) &
            (col_vuelta < n_pix_img)
        ]

        # --------------------------------------------------------
        # Índices de frame y línea dentro del frame
        # --------------------------------------------------------

        frame_idx = line_global // N_lineas
        y_idx = line_global % N_lineas

        if frame_idx >= n_frames:
            break

        # --------------------------------------------------------
        # Acumular fotones
        # --------------------------------------------------------

        if col_ida.size > 0:

            np.add.at(
                ida_stack[frame_idx, y_idx],
                col_ida,
                1
            )

        if col_vuelta.size > 0:

            np.add.at(
                vuelta_stack[frame_idx, y_idx],
                col_vuelta,
                1
            )

        n_lineas_procesadas += 1

    # ============================================================
    # 10) Coordenadas
    # ============================================================

    x = np.linspace(
        0,
        tamano_um,
        n_pix_img
    )

    y = np.linspace(
        0,
        tamano_um,
        N_lineas
    )

    # ============================================================
    # 11) Diagnóstico final
    # ============================================================

    print()
    print("Líneas detectadas:", n_lineas_totales)
    print("Líneas procesadas:", n_lineas_procesadas)
    print("Líneas sin fotones:", n_lineas_sin_fotones)
    print("Frames generados:", n_frames)
    print("Shape ida:", ida_stack.shape)
    print("Shape vuelta:", vuelta_stack.shape)
    print("================================")
    print()
    print(np.round(np.diff(truensync_mk) / dwell_sync, 2)[:20])
    # ------------------------------------------------------------
    # Líneas a partir de pares de markers (espaciado uniforme de 48)
    # ------------------------------------------------------------
    if truensync_mk.size % 2 == 1:
        truensync_mk = truensync_mk[:-1]      # 'end-1' suelto de la última línea
    
  
        



    return (
        x,
        y,
        ida_stack,
        vuelta_stack,
        n_frames
    )





if __name__ == "__main__":
    print("Version 1.0.00")

    path = r"C:\Users\Luis1\Downloads"
    file = "2-4"

    n_pix_img = 40
    n_pix_acc = 4
    N_lineas  = n_pix_img
    tamano_um = 2

    dwell_ns = 1000.0 * 1e3   # ej. 50 µs = 50e3 ns

    bin_width_ns = 0.032
    lifetime_ns = None
    #lifetime_ns  = (6,14)

    x, y, ida_stack, vuelta_stack, n_frames = imagen_ida_vuelta_desde_line_markers(
        path, file,
        n_pix_img=n_pix_img,
        n_pix_acc=n_pix_acc,
        tamano_um=tamano_um,
        dwell_ns=dwell_ns,
        N_lineas=N_lineas,
        lifetime_ns=lifetime_ns,
        bin_width_ns=bin_width_ns,
        N_delay = 2
    )

    # print(f"{n_frames} frames completos (líneas con markers inicio/fin)")
    
  
        

    # for f in range(0,len(ida_stack),1):
    #       graficar_ida(x, y, ida_stack[f], titulo="")
    #     graficar_vuelta(x, y, vuelta_stack[f], titulo=f"Vuelta frame {f}")
   
    frame = ida_stack[0]

    graficar_ida(x, y, frame, titulo="")

    # Ruta de salida
    archivo_tiff = os.path.join(path, f"{file}_ida_pic_frame.tiff")

    # Conversión a formato adecuado para TIFF
    frame = np.asarray(frame)

    if np.issubdtype(frame.dtype, np.floating):
        # Mantiene los valores normalizados en formato de 16 bits
        frame_min = frame.min()
        frame_max = frame.max()

        if frame_max > frame_min:
            frame_tiff = (
                (frame - frame_min) / (frame_max - frame_min) * 65535
            ).astype(np.uint32)
        else:
            frame_tiff = np.zeros_like(frame, dtype=np.uint16)
    else:
        frame_tiff = frame

    tifffile.imwrite(archivo_tiff, frame_tiff)

    print(f"Imagen guardada en: {archivo_tiff}")



 





