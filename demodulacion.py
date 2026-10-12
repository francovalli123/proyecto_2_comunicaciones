import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from configuracion import (
    INDICE_MODULACION_AM,
    FRECUENCIA_CORTE_AUDIO_HZ,
    AMPLITUD_PORTADORA,
    FRECUENCIAS_PORTADORAS_HZ,
    GUARDA_HZ,
    ORDEN_FILTRO,
    DESVIACION_FM_ANGOSTA_HZ,
    DESVIACION_FM_ANCHA_HZ,
    FS,
    CONSTANTE_DESVIACION_FASE_PM_RAD,
)

# Filtro pasabanda
def pasabanda(senial, f_c, ancho_banda):
    nyquist = FS / 2    # Frecuencia maxima que puede representarse sin distorsion a partir de las muestras
    bajo = max(10.0, f_c - ancho_banda)
    alto = min(nyquist - 10.0, f_c + ancho_banda)
    sos = butter(ORDEN_FILTRO, [bajo, alto], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, senial)

# Filtro pasabajo
def pasabajo(senial):
    sos = butter(
        ORDEN_FILTRO,
        FRECUENCIA_CORTE_AUDIO_HZ,
        btype="lowpass",
        fs=FS,
        output="sos",
    )
    return sosfiltfilt(sos, senial)

# Demodulacion en AM por detección coherente (porque conozco la portadora de antemano)
def demodular_am(recibida, f_c, ancho_banda, t):
    seleccionada = pasabanda(recibida, f_c, ancho_banda)    # Selecciona el canal ubicado alrededor de f_c    
    portadora_local = np.cos(2 * np.pi * f_c * t)
    banda_base = pasabajo(2 * seleccionada * portadora_local)   # Elimina la componente cerca de 2f_c

    # ELimino la componente de continua y deshago la escala dada por el indice de modulacion am
    mensaje = (banda_base - AMPLITUD_PORTADORA) / (AMPLITUD_PORTADORA * INDICE_MODULACION_AM)  

    return mensaje 

# Demodulacion en FM por derivación de fase instantánea
def demodular_fm(recibida, f_c, ancho_banda, f_d):
    x_canal = pasabanda(recibida, f_c, ancho_banda)    # Selecciono canal

    # Señal analítica + limitador
    z = hilbert(x_canal)
    z = z / np.abs(z)  # limitador de amplitud

    theta = np.unwrap(np.angle(z)) # Fase instantanea 
    f_inst = np.gradient(theta) / (2 * np.pi / FS) # Frecuencia instanteanea

    mensaje = (f_inst - f_c) / f_d # Extraer mensaje
    mensaje = pasabajo(mensaje)    # Limpiar ruido de alta frecuencia

    return mensaje

# Demodulación en PM
def demodular_pm(recibida, f_c, ancho_banda, k_p):
    x_canal = pasabanda(recibida, f_c, ancho_banda) # Selecciono canal

    # Señal analitica
    z = hilbert(x_canal)
    z = z / np.abs(z)

    theta = np.unwrap(np.angle(z))  # Fase instantanea
    # Quito la fase de la portadora
    t = np.arange(len(theta)) / FS 
    mensaje = (theta - 2 * np.pi * f_c * t) / k_p
    mensaje = pasabajo(mensaje)  # Limpio ruido de alta frecuencia

    return mensaje

# Estima cuando debe abarcar el filtro pasabanda alrededor de la frecuencia portadora para no eliminar informaicon del canal
def ancho_canal(esquema, canal, mensaje):
    pico_mensaje = float(np.max(np.abs(mensaje)))   # Amplitud maxima del mensaje
    if esquema == "AM":
        desviacion = 0.0    # En AM no hay desviacion de frecuencia
    elif esquema == "FM angosta":
        desviacion = DESVIACION_FM_ANGOSTA_HZ * pico_mensaje
    elif esquema == "FM ancha":
        desviacion = DESVIACION_FM_ANCHA_HZ * pico_mensaje
    else:
        derivada_mensaje = np.diff(mensaje) * FS
        desviacion = CONSTANTE_DESVIACION_FASE_PM_RAD * float(np.max(np.abs(derivada_mensaje))) / (2 * np.pi)
    
    limite_mensaje = 1_000 if canal == "tono" else FRECUENCIA_CORTE_AUDIO_HZ
    return desviacion + limite_mensaje + GUARDA_HZ

# Demodula todos los canales en todos los esquemas
def recuperar_seniales(transmisiones, mensajes, t):
    recuperadas = {}    # Diccionario
    for esquema, transmision in transmisiones.items():      # Recorre cada esquema de modulacion
        recuperadas[esquema] = {}
        for canal, mensaje in mensajes.items():     # Recorre cada canal
            portadora = FRECUENCIAS_PORTADORAS_HZ[canal]
            ancho_banda = ancho_canal(esquema, canal, mensaje)  # Calcula el ancho de banda necesario
            recibida = transmision["componer"]

            if esquema == "AM":
                salida = demodular_am(recibida, portadora, ancho_banda, t)
            elif esquema == "FM angosta":
                salida = demodular_fm(recibida, portadora, ancho_banda, DESVIACION_FM_ANGOSTA_HZ)
            elif esquema == "FM ancha":
                salida = demodular_fm(recibida, portadora, ancho_banda, DESVIACION_FM_ANCHA_HZ)
            else:
                salida = demodular_pm(recibida, portadora, ancho_banda, CONSTANTE_DESVIACION_FASE_PM_RAD)
            recuperadas[esquema][canal] = salida
    return recuperadas
