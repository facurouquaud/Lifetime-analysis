# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:54:11 2026

@author: Luis1
"""

from dataclasses import dataclass
from pathlib import Path
import struct
import sys

import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QMessageBox,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QFormLayout,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
)
from PyQt5.QtCore import QLocale

import read_PTU_pixels_2 as rd
import Analisis_lifetime_macro as alm
from dataclasses import dataclass

@dataclass
class PTUConfig:
    """
    Parámetros de reconstrucción de la imagen.
    """

    n_pix_img: int = 200
    n_pix_acc: int = 4
    n_lineas: int = n_pix_img
    tamano_um: float = 10.0
    dwell_us: float = 1200
    bin_width_ns: float = 0.032
    canal: int = 1





@dataclass
class PTUResult:
    """
    Resultado de la reconstrucción.
    """

    x: np.ndarray
    y: np.ndarray
    ida_stack: np.ndarray
    vuelta_stack: np.ndarray
    n_frames: int
    lifetime_ns: tuple
def leer_fotones_globales_ptu(archivo_ptu, apd_canal=1):
    """
    Lee todos los fotones del canal seleccionado.

    Devuelve:
        dtime_array
        truensync_array
        globRes
        timeRes
    """

    T3WRAPAROUND = 65536
    overflow = 0

    dtime_list = []
    truensync_list = []

    with open(archivo_ptu, "rb") as f:

        numRecords, globRes, timeRes = rd.readHeaders(f)

        for _ in range(numRecords):

            raw = f.read(4)

            if not raw:
                break

            recordData = struct.unpack("<I", raw)[0]

            channel = recordData >> 28
            dtime = (recordData >> 16) & 0xFFF
            nsync = recordData & 0xFFFF

            if channel == 15:

                if dtime == 0:
                    overflow += T3WRAPAROUND

            elif channel == apd_canal:

                truensync = overflow + nsync

                dtime_list.append(dtime)
                truensync_list.append(truensync)

    dtime_array = np.asarray(dtime_list, dtype=np.int32)
    truensync_array = np.asarray(
        truensync_list,
        dtype=np.int64
    )

    lifetime_ns = dtime_array * timeRes * 1e9

    return {
        "dtime": dtime_array,
        "truensync": truensync_array,
        "lifetime_ns": lifetime_ns,
        "globRes": globRes,
        "timeRes": timeRes,
        "numRecords": numRecords,
    }
def obtener_histograma_lifetime(lifetime_ns, bins=100):
    """
    Calcula el histograma de lifetime.
    """

    if lifetime_ns.size == 0:
        return np.array([]), np.array([])

    counts, edges = np.histogram(
        lifetime_ns,
        bins=bins
    )

    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, counts
def reconstruir_imagen_ptu(
    archivo_ptu,
    config,
    lifetime_window=None
):
    archivo_ptu = Path(archivo_ptu)

    path = str(archivo_ptu.parent)
    file = archivo_ptu.stem

    resultado = alm.imagen_ida_vuelta_desde_line_markers(
        path=path,
        file=file,
        n_pix_img=config.n_pix_img,
        n_pix_acc=config.n_pix_acc,
        tamano_um=config.tamano_um,
        dwell_ns=config.dwell_us*1000,
        N_lineas=config.n_lineas,
        lifetime_ns=lifetime_window,
        bin_width_ns=config.bin_width_ns
    )

    x, y, ida_stack, vuelta_stack, n_frames = resultado

    return PTUResult(
        x=x,
        y=y,
        ida_stack=ida_stack,
        vuelta_stack=vuelta_stack,
        n_frames=n_frames,
        lifetime_ns=(
            lifetime_window
            if lifetime_window is not None
            else (None, None)
        )
    )

def calcular_frc(imagen_a, imagen_b, n_anillos=None):
    """
    Calcula una curva FRC básica.
    """

    imagen_a = np.asarray(imagen_a, dtype=float)
    imagen_b = np.asarray(imagen_b, dtype=float)

    if imagen_a.shape != imagen_b.shape:
        raise ValueError(
            "Las imágenes deben tener la misma dimensión."
        )

    imagen_a = imagen_a - np.mean(imagen_a)
    imagen_b = imagen_b - np.mean(imagen_b)

    fft_a = np.fft.fftshift(
        np.fft.fft2(imagen_a)
    )

    fft_b = np.fft.fftshift(
        np.fft.fft2(imagen_b)
    )

    alto, ancho = imagen_a.shape

    fy = np.fft.fftshift(np.fft.fftfreq(alto))
    fx = np.fft.fftshift(np.fft.fftfreq(ancho))

    fx_grid, fy_grid = np.meshgrid(fx, fy)

    radio = np.sqrt(
        fx_grid**2 + fy_grid**2
    )

    if n_anillos is None:
        n_anillos = min(imagen_a.shape) // 2

    radio_max = np.max(radio)
    bordes = np.linspace(
        0,
        radio_max,
        n_anillos + 1
    )

    frecuencias = []
    valores_frc = []

    for i in range(n_anillos):

        mascara = (
            (radio >= bordes[i]) &
            (radio < bordes[i + 1])
        )

        if not np.any(mascara):
            continue

        a = fft_a[mascara]
        b = fft_b[mascara]

        numerador = np.sum(
            a * np.conjugate(b)
        )

        denominador = np.sqrt(
            np.sum(np.abs(a) ** 2) *
            np.sum(np.abs(b) ** 2)
        )

        if denominador == 0:
            frc = 0.0
        else:
            frc = np.real(
                numerador / denominador
            )

        frecuencias.append(
            0.5 * (bordes[i] + bordes[i + 1])
        )

        valores_frc.append(frc)

    return (
        np.asarray(frecuencias),
        np.asarray(valores_frc)
    )
class LifetimeApp(QMainWindow):

    def __init__(self):
        super().__init__()
    
        self.setWindowTitle(
            "PicoHarp 300 — Lifetime Imaging"
        )
    
        self.resize(1500, 850)
    
        self.archivo_ptu = None
        self.config = PTUConfig()
        self.resultado = None
        self.hist_lifetime = None
    
        self.t_min_ns = 0.0
        self.t_max_ns = 25.0
    
        self._crear_interfaz()
        self._aplicar_estilo()


    # -----------------------------------------------------
    # Construcción de la interfaz
    # -----------------------------------------------------

    def _crear_interfaz(self):

        central = QWidget()
        self.setCentralWidget(central)

        layout_principal = QHBoxLayout(central)
        layout_principal.setContentsMargins(12, 12, 12, 12)
        layout_principal.setSpacing(12)

        panel_controles = self._crear_panel_controles()
        panel_imagenes = self._crear_panel_imagenes()
        panel_graficos = self._crear_panel_graficos()

        layout_principal.addLayout(
            panel_controles,
            stretch=1
        )

        layout_principal.addLayout(
            panel_imagenes,
            stretch=4
        )

        layout_principal.addLayout(
            panel_graficos,
            stretch=2
        )

    def _crear_panel_controles(self):

        panel = QVBoxLayout()
        panel.setSpacing(10)

        grupo_archivo = QGroupBox("Archivo PTU")
        layout_archivo = QVBoxLayout()

        self.boton_cargar = QPushButton(
            "Cargar archivo PTU"
        )

        self.boton_cargar.clicked.connect(
            self.cargar_archivo
        )

        self.label_archivo = QLabel(
            "Ningún archivo cargado"
        )

        self.label_archivo.setWordWrap(True)

        self.label_info = QLabel(
            "Fotones: —"
        )

        self.label_info.setWordWrap(True)

        layout_archivo.addWidget(
            self.boton_cargar
        )

        layout_archivo.addWidget(
            self.label_archivo
        )

        layout_archivo.addWidget(
            self.label_info
        )

        grupo_archivo.setLayout(
            layout_archivo
        )

        grupo_parametros = QGroupBox(
            "Parámetros de reconstrucción"
        )

        form_parametros = QFormLayout()

        self.spin_npix = QSpinBox()
        self.spin_npix.setRange(4, 2048)
        self.spin_npix.setValue(
            self.config.n_pix_img
        )

        self.spin_nlineas = QSpinBox()
        self.spin_nlineas.setRange(4, 2048)
        self.spin_nlineas.setValue(
            self.config.n_lineas
        )

        self.spin_acc = QSpinBox()
        self.spin_acc.setRange(0, 100)
        self.spin_acc.setValue(
            self.config.n_pix_acc
        )

        self.spin_tamano = QDoubleSpinBox()
        self.spin_tamano.setRange(0.001, 10000)
        self.spin_tamano.setDecimals(3)
        self.spin_tamano.setValue(
            self.config.tamano_um
        )
        self.spin_dwell = QDoubleSpinBox()
        
        self.spin_dwell.setLocale(
            QLocale(QLocale.Spanish, QLocale.Spain)
        )
        self.spin_dwell.setDecimals(1)
        self.spin_dwell.setSingleStep(0.1)
        self.spin_dwell.setRange(0.0, 10000.0)
        self.spin_dwell.setValue(self.config.dwell_us)
        self.spin_dwell.setSuffix(" µs")



        self.spin_bin = QDoubleSpinBox()
        self.spin_bin.setRange(1e-6, 1000)
        self.spin_bin.setDecimals(6)
        self.spin_bin.setValue(
            self.config.bin_width_ns
        )

        self.combo_canal = QComboBox()
        self.combo_canal.addItems(["1", "2"])
        self.combo_canal.setCurrentText(
            str(self.config.canal)
        )

        form_parametros.addRow(
            "Píxeles útiles:",
            self.spin_npix
        )



        form_parametros.addRow(
            "Píxeles aceleración:",
            self.spin_acc
        )

        form_parametros.addRow(
            "Tamaño imagen (µm):",
            self.spin_tamano
        )

        form_parametros.addRow(
            "Dwell (µs):",
            self.spin_dwell
        )

        form_parametros.addRow(
            "Bin lifetime (µs):",
            self.spin_bin
        )

        form_parametros.addRow(
            "Canal APD:",
            self.combo_canal
        )

        grupo_parametros.setLayout(
            form_parametros
        )

        grupo_lifetime = QGroupBox(
            "Ventana de lifetime"
        )

        form_lifetime = QFormLayout()

        self.spin_min = QDoubleSpinBox()
        self.spin_min.setRange(-10000, 10000)
        self.spin_min.setDecimals(1)
        self.spin_min.setValue(8.5)

        self.spin_max = QDoubleSpinBox()
        self.spin_max.setRange(-10000, 10000)
        self.spin_max.setDecimals(1)
        self.spin_max.setValue(13.0)

        self.spin_min.valueChanged.connect(
            self.actualizar_ventana
        )

        self.spin_max.valueChanged.connect(
            self.actualizar_ventana
        )

        self.label_ventana = QLabel(
            "Ventana: —"
        )

        form_lifetime.addRow(
            "Mínimo (ns):",
            self.spin_min
        )

        form_lifetime.addRow(
            "Máximo (ns):",
            self.spin_max
        )

        form_lifetime.addRow(
            self.label_ventana
        )

        grupo_lifetime.setLayout(
            form_lifetime
        )

        self.combo_modo = QComboBox()
        self.combo_modo.addItems(["Ida", "Vuelta"])
        self.combo_modo.setCurrentText("Ida")

        self.combo_modo.currentIndexChanged.connect(
            self.actualizar_imagen_mostrada
        )

        self.boton_reconstruir = QPushButton(
            "Reconstruir imagen"
        )

        self.boton_reconstruir.clicked.connect(
            self.reconstruir
        )

        self.boton_frc = QPushButton(
            "Calcular FRC ida/vuelta"
        )

        self.boton_frc.clicked.connect(
            self.calcular_frc
        )

        self.boton_guardar = QPushButton(
            "Guardar imagen"
        )

        self.boton_guardar.clicked.connect(
            self.guardar_imagen
        )

        self.boton_reconstruir.setEnabled(False)
        self.boton_frc.setEnabled(False)
        self.boton_guardar.setEnabled(False)

        panel.addWidget(grupo_archivo)
        panel.addWidget(grupo_parametros)
        panel.addWidget(grupo_lifetime)
        panel.addWidget(QLabel("Imagen a mostrar:"))
        panel.addWidget(self.combo_modo)
        panel.addWidget(self.boton_reconstruir)
        panel.addWidget(self.boton_frc)
        panel.addWidget(self.boton_guardar)
        panel.addStretch()

        return panel

    def _crear_panel_imagenes(self):

        panel = QVBoxLayout()

        self.imagen_view = pg.ImageView()
        self.imagen_view.ui.roiBtn.hide()
        self.imagen_view.ui.menuBtn.hide()

        self.imagen_view.setSizePolicy(
            QWidget().sizePolicy()
        )

        panel.addWidget(
            QLabel("Imagen reconstruida")
        )

        panel.addWidget(
            self.imagen_view
        )

        return panel

    def _crear_panel_graficos(self):

        panel = QVBoxLayout()

        self.histograma = pg.PlotWidget()
        self.histograma.setLabel(
            "bottom",
            "Lifetime",
            units="ns"
        )

        self.histograma.setLabel(
            "left",
            "Cuentas"
        )

        self.histograma.showGrid(
            x=True,
            y=True,
            alpha=0.25
        )

        self.frc_plot = pg.PlotWidget()
        self.frc_plot.setLabel(
            "bottom",
            "Frecuencia espacial"
        )

        self.frc_plot.setLabel(
            "left",
            "FRC"
        )

        self.frc_plot.showGrid(
            x=True,
            y=True,
            alpha=0.25
        )

        panel.addWidget(
            QLabel("Histograma de lifetime")
        )

        panel.addWidget(
            self.histograma,
            stretch=2
        )

        panel.addWidget(
            QLabel("Fourier Ring Correlation")
        )

        panel.addWidget(
            self.frc_plot,
            stretch=1
        )

        return panel

    # -----------------------------------------------------
    # Estética
    # -----------------------------------------------------

    def _aplicar_estilo(self):

        self.setStyleSheet("""
            QMainWindow {
                background-color: #20242b;
            }

            QWidget {
                color: #e8ecf1;
                font-size: 13px;
            }

            QGroupBox {
                border: 1px solid #424955;
                border-radius: 7px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #8fc7ff;
            }

            QPushButton {
                background-color: #315d85;
                border: none;
                border-radius: 5px;
                padding: 8px;
            }

            QPushButton:hover {
                background-color: #3d76a8;
            }

            QPushButton:disabled {
                background-color: #444a53;
                color: #8a9099;
            }

            QDoubleSpinBox,
            QSpinBox,
            QComboBox {
                background-color: #2d323a;
                border: 1px solid #505762;
                border-radius: 4px;
                padding: 4px;
            }

            QLabel {
                color: #d9dde4;
            }
        """)

    # -----------------------------------------------------
    # Lectura
    # -----------------------------------------------------

    def cargar_archivo(self):

        nombre, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo PTU",
            "",
            "Archivos PicoQuant (*.ptu)"
        )

        if not nombre:
            return

        try:

            self.archivo_ptu = nombre

            datos = leer_fotones_globales_ptu(
                nombre,
                apd_canal=int(
                    self.combo_canal.currentText()
                )
            )

            self.hist_lifetime = datos["lifetime_ns"]

            self.label_archivo.setText(
                Path(nombre).name
            )

            self.label_info.setText(
                f"Fotones: {len(self.hist_lifetime):,}\n"
                f"Lifetime mínimo: "
                f"{np.min(self.hist_lifetime):.4f} ns\n"
                f"Lifetime máximo: "
                f"{np.max(self.hist_lifetime):.4f} ns\n"
                f"Resolución temporal: "
                f"{datos['timeRes'] * 1e9:.6f} ns"
            )

            self.spin_min.setValue(
                float(np.percentile(
                    self.hist_lifetime,
                    10
                ))
            )

            self.spin_max.setValue(
                float(np.percentile(
                    self.hist_lifetime,
                    90
                ))
            )

            self.mostrar_histograma()

            self.boton_reconstruir.setEnabled(True)

        except Exception as error:

            QMessageBox.critical(
                self,
                "Error al cargar PTU",
                str(error)
            )

    # -----------------------------------------------------
    # Histograma
    # -----------------------------------------------------

    def mostrar_histograma(self):

        self.histograma.clear()

        if self.hist_lifetime is None:
            return

        centers, counts = obtener_histograma_lifetime(
            self.hist_lifetime,
            bins=100
        )

        curva = pg.PlotCurveItem(
            centers,
            counts,
            pen=pg.mkPen(
                "#8fc7ff",
                width=2
            )
        )

        self.histograma.addItem(curva)

        self.linea_min = pg.InfiniteLine(
            pos=self.spin_min.value(),
            angle=90,
            pen=pg.mkPen(
                "#ff9f43",
                width=2
            )
        )

        self.linea_max = pg.InfiniteLine(
            pos=self.spin_max.value(),
            angle=90,
            pen=pg.mkPen(
                "#ff9f43",
                width=2
            )
        )

        self.histograma.addItem(self.linea_min)
        self.histograma.addItem(self.linea_max)

    def actualizar_ventana(self):

        minimo = self.spin_min.value()
        maximo = self.spin_max.value()
    
        if maximo <= minimo:
            self.label_ventana.setText(
                "Ventana no válida: el máximo debe ser mayor que el mínimo"
            )
            return
    
        self.label_ventana.setText(
            f"Ventana: {minimo:.4f} – {maximo:.4f} ns"
        )
    
        if hasattr(self, "linea_min"):
            self.linea_min.setPos(minimo)
            self.linea_max.setPos(maximo)


    # -----------------------------------------------------
    # Reconstrucción
    # -----------------------------------------------------

    def obtener_configuracion(self):

        return PTUConfig(
            n_pix_img=self.spin_npix.value(),
            n_pix_acc=self.spin_acc.value(),
            n_lineas=self.spin_nlineas.value(),
            tamano_um=self.spin_tamano.value(),
            dwell_us=self.spin_dwell.value(),
            bin_width_ns=self.spin_bin.value(),
            canal=int(
                self.combo_canal.currentText()
            )
        )

    def reconstruir(self):

        if self.archivo_ptu is None:
            return
    
        minimo = self.spin_min.value()
        maximo = self.spin_max.value()
    
        if maximo <= minimo:
            QMessageBox.warning(
                self,
                "Ventana de lifetime no válida",
                "El máximo debe ser mayor que el mínimo."
            )
            return
    
        try:
            self.config = self.obtener_configuracion()
    
            ventana = (minimo, maximo)
    
            self.resultado = reconstruir_imagen_ptu(
                archivo_ptu=self.archivo_ptu,
                config=self.config,
                lifetime_window=ventana
            )
    
            self.actualizar_imagen_mostrada()
    
            self.boton_frc.setEnabled(True)
            self.boton_guardar.setEnabled(True)
    
        except Exception as error:
            QMessageBox.critical(
                self,
                "Error durante la reconstrucción",
                str(error)
            )


    def actualizar_imagen_mostrada(self):

        if self.resultado is None:
            return

        ida = self.resultado.ida_stack[0].astype(float)
        vuelta = self.resultado.vuelta_stack[0].astype(float)

        modo = self.combo_modo.currentText()

        if modo == "Ida":
            imagen = np.flip(ida.T, axis = 1)

        elif modo == "Vuelta":
            imagen = np.fliplr(vuelta)

        else:
            imagen = ida + np.fliplr(vuelta)

        self.imagen_view.setImage(
            imagen,
            autoLevels=True
        )

    # -----------------------------------------------------
    # FRC
    # -----------------------------------------------------

    def calcular_frc(self):

        if self.resultado is None:
            return

        ida = self.resultado.ida_stack[0].astype(float)
        vuelta = np.fliplr(
            self.resultado.vuelta_stack[0].astype(float)
        )

        frecuencias, valores = calcular_frc(
            ida,
            vuelta
        )

        self.frc_plot.clear()

        self.frc_plot.plot(
            frecuencias,
            valores,
            pen=pg.mkPen(
                "#7bed9f",
                width=2
            )
        )

        umbral = pg.InfiniteLine(
            pos=1 / 7,
            angle=0,
            pen=pg.mkPen(
                "#ff6b6b",
                style=Qt.DashLine
            )
        )

        self.frc_plot.addItem(umbral)
        self.frc_plot.setYRange(0, 1.05)

    # -----------------------------------------------------
    # Guardado
    # -----------------------------------------------------

    def guardar_imagen(self):

        if self.resultado is None:
            return

        nombre, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar imagen",
            "",
            "TIFF (*.tif *.tiff);;NumPy (*.npy)"
        )

        if not nombre:
            return

        ida = self.resultado.ida_stack[0].astype(float)
        vuelta = np.fliplr(
            self.resultado.vuelta_stack[0].astype(float)
        )

        modo = self.combo_modo.currentText()

        if modo == "Ida":
            imagen = ida

        elif modo == "Vuelta":
            imagen = vuelta

        else:
            imagen = ida + vuelta

        if nombre.lower().endswith(".npy"):
            np.save(nombre, imagen)

        else:
            imagen_max = np.max(imagen)

            if imagen_max > 0:
                imagen_guardar = (
                    imagen / imagen_max * 65535
                ).astype(np.uint16)
            else:
                imagen_guardar = imagen.astype(np.uint16)

            import tifffile

            tifffile.imwrite(
                nombre,
                imagen_guardar
            )
if __name__ == "__main__":

    app = QApplication(sys.argv)

    ventana = LifetimeApp()
    ventana.show()

    sys.exit(app.exec())
