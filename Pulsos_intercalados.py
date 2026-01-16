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
def read_phu(file, time_res=32e-12):
    """
    Lector simple de archivos PHU (PQHISTO).
    La resolución temporal se fuerza (default: 32 ps).
    """

    import struct
    import numpy as np
    import os
    from pathlib import Path

    close_after = False
    if isinstance(file, (str, Path)):
        f = open(file, "rb")
        close_after = True
        filename = file
    else:
        f = file
        filename = None

    try:
        # --- Header ---
        magic = f.read(8).decode(errors="ignore").strip()
        if not magic.startswith("PQHISTO"):
            raise ValueError(f"Archivo PHU inválido (magic = {magic})")

        f.read(8)  # versión (no la usamos)

        tags = {}
        while True:
            tagIdent = f.read(32).decode(errors="ignore").strip("\x00")
            tagIdx   = struct.unpack("<i", f.read(4))[0]
            tagType  = struct.unpack("<i", f.read(4))[0]
            tagValue = f.read(8)

            if tagType == 0xFFFFFFFF:
                value = None
            elif tagType in (0x00000001, 0x00000002):
                value = struct.unpack("<i", tagValue)[0]
            elif tagType == 0x00000003:
                value = struct.unpack("<d", tagValue)[0]
            elif tagType == 0x00000004:
                pos = struct.unpack("<q", tagValue)[0]
                cur = f.tell()
                f.seek(pos)
                value = f.read(256).decode(errors="ignore").strip("\x00")
                f.seek(cur)
            else:
                value = tagValue

            tags[(tagIdent, tagIdx)] = value
            if tagIdent == "Header_End":
                break

        data_start = f.tell()

        # --- Leer histograma ---
        if filename:
            file_size = os.path.getsize(filename)
        else:
            cur = f.tell()
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(cur)

        n_bins = (file_size - data_start) // 4

        f.seek(data_start)
        hist = np.fromfile(f, dtype=np.uint32, count=n_bins)

        return hist, time_res, tags

    finally:
        if close_after:
            f.close()



#%%
hist, time_res, tags = read_phu(
    r"C:\Users\Luis1\Downloads\pulso_rojo_25_50.phu"
)

t_ns = np.arange(len(hist)) * time_res * 1e9/2

plt.figure(figsize=(6,4))
plt.plot(t_ns, hist)
plt.xlabel("Tiempo dentro del período (ns)")
plt.ylabel("Cuentas")
plt.title("IRF plegada al período del láser")
plt.xlim(0, 60)
plt.show()
#%%

n_rep = 3  # repetir 3 períodos
hist_rep = np.tile(hist, n_rep)
t_rep = np.arange(len(hist_rep)) * time_res * 1e9

plt.plot(t_rep, hist_rep)
plt.xlabel("Tiempo (ns)")
plt.ylabel("Cuentas")
plt.title("Visualización extendida (como el software)")
plt.show()