import numpy as np
import soundfile as sf
from math import gcd
from scipy.signal import butter, resample_poly, sosfiltfilt

from configuracion import (
    AMPLITUD_PORTADORA,
    FRECUENCIA_CORTE_AUDIO_HZ,
    FS,
    INDICE_MODULACION_AM,
    INPUT_AUDIO,
)

# Funcion que carga el audio 
def cargar_audio():
    audio, fs_original = sf.read(INPUT_AUDIO, dtype="float64")   # Lee las muestras como flotantes y guarda la frecuencia original del archivo

    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # Como son dos canales, promedia los canales y crea una sola señal mono

    # Solo remuestrea cuando el archivo no tiene la frecuencia usada por los moduladores.
    if fs_original != FS:
        # Reduce la proporcion FS/fs_original a dos factores enteros equivalentes.
        divisor = gcd(FS, int(fs_original))
        # El primer factor aumenta la tasa y el segundo la reduce hasta llegar a FS.
        audio = resample_poly(
            audio,
            FS // divisor,
            int(fs_original) // divisor,
        )

    # Limita el ancho de banda del audio antes de transmitirlo.
    sos = butter(6, FRECUENCIA_CORTE_AUDIO_HZ, btype="lowpass", fs=FS, output="sos")
    audio = sosfiltfilt(sos, audio)

    audio_referencia = audio.copy()

    mensaje = audio.copy()    # Trabaja sobre otra copia para conservar intacta la referencia.
    mensaje -= mensaje.mean()
    pico = np.max(np.abs(mensaje))
    if pico > 0:
        mensaje *= 0.9 / pico    # Si hay señal, ajusta el pico a 0.9, si es silencio, evita dividir por cero.

    # Calcula la envolvente teorica de AM para cada muestra del mensaje.
    envolvente = AMPLITUD_PORTADORA * (
        1 + INDICE_MODULACION_AM * mensaje
    )


    # Asocia cada muestra con su instante usando el periodo de muestreo 1 / FS.
    t = np.arange(mensaje.size) / FS
    return mensaje, audio_referencia, t
