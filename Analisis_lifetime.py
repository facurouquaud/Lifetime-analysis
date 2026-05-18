import read_PTU_pixels_2 as rd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def graficar_ida(x, y, imagen):
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def graficar_vuelta(x, y, imagen):
    imagen = np.flip(imagen, axis=1)
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def separar_ida_vuelta(pixeles_ida, pixeles_vuelta, shape, n_pix_objetivo, canal):
    cuentas_ida = np.array([len(p[canal]) for p in pixeles_ida], dtype=np.int32)
    cuentas_vuelta = np.array([len(p[canal]) for p in pixeles_vuelta], dtype=np.int32)

    if len(cuentas_ida) < n_pix_objetivo:
        cuentas_ida = np.pad(cuentas_ida, (0, n_pix_objetivo - len(cuentas_ida)))
    else:
        cuentas_ida = cuentas_ida[:n_pix_objetivo]

    if len(cuentas_vuelta) < n_pix_objetivo:
        cuentas_vuelta = np.pad(cuentas_vuelta, (0, n_pix_objetivo - len(cuentas_vuelta)))
    else:
        cuentas_vuelta = cuentas_vuelta[:n_pix_objetivo]

    return cuentas_ida.reshape(shape), cuentas_vuelta.reshape(shape)

# --------- multi-frame basado en filtrar_pixels ---------

def imagen_ida_vuelta_por_bloques(path, file,
                                  n_pix_img,     # 200
                                  n_pix_acc,     # 4
                                  tamano_um,
                                  pixeles_ida_centro,
                                  N_lineas,      # 200 líneas por frame
                                  n_pix_gap,     # píxeles CRUDOS a saltear entre frames (200)
                                  canal=1):

    archivo = f"{path}\\{file}.ptu"

    # Leer eventos crudos
    with open(archivo, 'rb') as fd:
        numRecords, _, _ = rd.readHeaders(fd)
        _, _, pixeles = rd.readPT3_fast_pixels(fd, numRecords)

    # Recorte al inicio en datos crudos (calibrado por vos)
    pixeles = pixeles[pixeles_ida_centro:]
    total = len(pixeles)

    # Longitud cruda de UNA línea
    puntos_por_linea = 2 * n_pix_img + 4 * n_pix_acc

    # Longitud cruda de UN FRAME completo (todas las líneas)
    raw_frame = N_lineas * puntos_por_linea

    # Bloque total crudo por frame: frame completo + n_pix_gap entre frames
    bloque = raw_frame + n_pix_gap

    

    # Máximo número de frames tales que:
    # start_f + raw_frame <= total, con start_f = f*bloque
    n_frames = (total - raw_frame) // bloque + 1

    shape = (n_pix_img, n_pix_img)
    n_pix_objetivo = shape[0] * shape[1]

    ida_stack    = np.zeros((n_frames, n_pix_img, n_pix_img), dtype=np.int32)
    vuelta_stack = np.zeros((n_frames, n_pix_img, n_pix_img), dtype=np.int32)

    frames_validos = 0

    for f in range(n_frames):
        start_block = f * bloque
        end_utile   = start_block + raw_frame  # todo el escaneo de un frame

        if end_utile > total:
            break  # último frame incompleto

        subpix = pixeles[start_block:end_utile]

        # AHORA SÍ: N = N_lineas (todas las líneas del frame)
        _, pix_ida, pix_vuelta = rd.filtrar_pixels(
            subpix,
            n_pix_img,
            n_pix_acc,
            N_lineas
        )

        imagen_ida, imagen_vuelta = separar_ida_vuelta(
            pix_ida, pix_vuelta, shape, n_pix_objetivo, canal=canal
        )

        ida_stack[frames_validos]    = imagen_ida
        vuelta_stack[frames_validos] = imagen_vuelta
        frames_validos += 1

    ida_stack    = ida_stack[:frames_validos]
    vuelta_stack = vuelta_stack[:frames_validos]
    n_frames     = frames_validos

    x = np.linspace(0, tamano_um, n_pix_img)
    y = np.linspace(0, tamano_um, n_pix_img)

    return x, y, ida_stack, vuelta_stack, n_frames



def graficar_frame_ida_vuelta(x, y, ida_stack, vuelta_stack, frame_idx):
    graficar_ida(x, y, ida_stack[frame_idx])
    graficar_vuelta(x, y, vuelta_stack[frame_idx])


def intensidad_roi_en_el_tiempo(stack, y_min, y_max, x_min, x_max):
    """
    stack: array (n_frames, n_pix, n_pix) con número de fotones (ida o vuelta)
    ROI:   índices de píxel [y_min:y_max, x_min:x_max] (inclusive en min, exclusivo en max)
    Devuelve:
        t_index:   np.arange(n_frames)
        intensidad: suma de fotones en ROI para cada frame
    """
    n_frames = stack.shape[0]

    # Recortar ROI (y primero, luego x)
    roi = stack[:, y_min:y_max, x_min:x_max]   # -> (n_frames, dy, dx)

    # Sumar fotones en la ROI por frame
    intensidad = roi.sum(axis=(1, 2))         # -> (n_frames,)

    t_index = np.arange(1, n_frames + 1)
    return t_index, intensidad

def graficar_traza_roi(t_index, intensidad, tiempo_por_frame=None, titulo="ROI"):
    """
    Si tiempo_por_frame (en ms, s, etc.) es None, el eje x es el número de frame.
    Si se da tiempo_por_frame, x = t_index * tiempo_por_frame.
    """
    if tiempo_por_frame is None:
        x = t_index
        xlabel = "Frame"
    else:
        x = t_index * tiempo_por_frame
        xlabel = "Tiempo"

    plt.figure(constrained_layout=True)
    plt.plot(x, intensidad, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Fotones en ROI")
    plt.title(f"Intensidad vs tiempo ({titulo})")
    plt.grid(True)
    plt.show()
def mostrar_frame_con_roi(x, y, imagen, y_min, y_max, x_min, x_max,
                          color='red', linewidth=2):
    """
    Dibuja un rectángulo que marca la ROI sobre una imagen (un frame del stack).
    ROI en índices de píxel: [y_min:y_max, x_min:x_max]
    """
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')

    # Convertir índices de píxel a coordenadas físicas (µm)
    # Ojo: extent = [x0, x1, y0, y1] y origin='lower'
    dx = (x.max() - x.min()) / imagen.shape[1]
    dy = (y.max() - y.min()) / imagen.shape[0]

    x0 = x.min() + x_min * dx
    y0 = y.min() + y_min * dy
    width  = (x_max - x_min) * dx
    height = (y_max - y_min) * dy

    rect = Rectangle((x0, y0), width, height,
                     linewidth=linewidth, edgecolor=color,
                     facecolor='none')
    ax.add_patch(rect)

    plt.show()




def seleccionar_roi_con_ginput(x, y, imagen):
    """
    Muestra la imagen y permite seleccionar una ROI con dos clics:
    primer clic = esquina inferior izquierda,
    segundo clic = esquina superior derecha (en coordenadas físicas).

    Devuelve (y_min, y_max, x_min, x_max) en índices de píxel,
    o (None, None, None, None) si no se seleccionó nada.
    """
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')

    print("Hacé DOS clics: esquina 1 y esquina 2 de la ROI, luego cerrá la ventana.")

    # ginput devuelve una lista de puntos (x,y) en coordenadas de datos
    pts = plt.ginput(2, timeout=-1)  # espera hasta que hagas 2 clics
    plt.close(fig)

    if len(pts) < 2:
        return None, None, None, None

    (x0, y0), (x1, y1) = pts

    ny, nx = imagen.shape
    dx = (x.max() - x.min()) / nx
    dy = (y.max() - y.min()) / ny

    ix0 = int((min(x0, x1) - x.min()) / dx)
    ix1 = int((max(x0, x1) - x.min()) / dx)
    iy0 = int((min(y0, y1) - y.min()) / dy)
    iy1 = int((max(y0, y1) - y.min()) / dy)


    return iy0, iy1, ix0, ix1

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
        imagen_ida, imagen_vuelta = separar_ida_vuelta(
           pixeles_ida, pixeles_vuelta, shape, n_pix_objetivo, canal=1)
        return x, y, imagen_ida, imagen_vuelta  


if __name__ == "__main__":
    path = r"C:\Users\Luis1\Downloads"
    file = "\prueba_sted_sm"

    n_pix_img = 200
    n_pix_acc = 4
    N_lineas  = 200
    tamano_um = 5
    pixeles_ida_centro = 72  # recorte inicial de pixeles de ida al frame

    n_pix_gap = 128    #  píxeles entre frames

    x, y, ida_stack, vuelta_stack, n_frames = imagen_ida_vuelta_por_bloques(
        path=path,
        file=file,
        n_pix_img=n_pix_img,
        n_pix_acc=n_pix_acc,
        tamano_um=tamano_um,
        pixeles_ida_centro=pixeles_ida_centro,
        N_lineas=N_lineas,
        n_pix_gap=n_pix_gap,
        canal=1
    )

    print(f"{n_frames} frames completos")
    
    # #Acá podemos ver los stcaks
   
    graficar_ida(x, y, ida_stack[0])
    graficar_vuelta(x, y, vuelta_stack[0])
    # number_of_pixels = 200
    # px_size  = 5/200
    # image_size_um = 10
    # pixeles_ida_al_cero = 72
    # dwell_time = 60
    # x, y, imagen_ida, imagen_vuelta = imagen_ida_vuelta(file, number_of_pixels,
    # image_size_um, pixeles_ida_al_cero)
    # graficar_ida(x,y,imagen_ida)
    # graficar_vuelta(x,y,imagen_vuelta)
   
    
    # #Con esto agarramos las coordenadas de los píxeles que tienen emisores
    
    # # frame_idx = 0  # frame donde querés elegir la ROI
    # # y_min, y_max, x_min, x_max = seleccionar_roi_con_ginput(
    # #     x, y, ida_stack[frame_idx]
    # # )
    
    # # if y_min is None:
    # #     print("No se seleccionó ninguna ROI. Saltando análisis de ROI.")
    # # else:
    # #     print("ROI en píxeles:", y_min, y_max, x_min, x_max)
    
    # #     # Calcular traza temporal en esa ROI
    # #     t_index, intensidad_ida = intensidad_roi_en_el_tiempo(
    # #         ida_stack, y_min, y_max, x_min, x_max
    # #     )
    # #     graficar_traza_roi(t_index, intensidad_ida,
    # #                         tiempo_por_frame=None,
    # #                         titulo=f"ROI ({y_min}:{y_max}, {x_min}:{x_max})")
    
    
    # #Con esto hacemos la traza y vemos donde esta el ROI
    
    # #traza de intensidad
    # y_min, y_max = 183, 203  # filas
    # x_min, x_max = 0, 20 # columnas
   
    # t_index, intensidad_ida = intensidad_roi_en_el_tiempo(
    #     ida_stack, y_min, y_max, x_min, x_max
    # )
   
    # # Si sabés el tiempo por frame (en µs, ms, s, etc.), ponelo aquí
    # tiempo_por_frame = None  # por ahora solo frame index
   
    # graficar_traza_roi(t_index, intensidad_ida,
    #                     tiempo_por_frame=tiempo_por_frame,
    #                     titulo="ROI (ida)")
    # # Mostrar, por ejemplo, el frame 0 (ida)
    # frame_idx = 0
    # mostrar_frame_con_roi(x, y, ida_stack[frame_idx],
    #                       y_min, y_max, x_min, x_max)
    
    

