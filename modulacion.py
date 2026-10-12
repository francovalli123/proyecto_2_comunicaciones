import numpy as np
from scipy.integrate import cumulative_trapezoid


from configuracion import (
    INDICE_MODULACION_AM,
    AMPLITUD_PORTADORA,
    FRECUENCIAS_PORTADORAS_HZ,
    FS,
    DESVIACION_FM_ANGOSTA_HZ,
    DESVIACION_FM_ANCHA_HZ,
    CONSTANTE_DESVIACION_FASE_PM_RAD,
    ESQUEMAS,
)

# Modulador AM
def modular_am(mensaje, f_c, t):
    mensaje_centrado = mensaje - np.mean(mensaje)   # Quito la componente de continua de m(t)
    mensaje_modulador = INDICE_MODULACION_AM * mensaje_centrado
    portadora = np.cos(2 * np.pi * f_c * t)
    x_am = AMPLITUD_PORTADORA * (1 + mensaje_modulador) * portadora 
    return x_am, mensaje_modulador

# Modulador FM
def modular_fm(mensaje, f_c, f_d, t):
    mensaje_centrado = mensaje - np.mean(mensaje)   # Quito la componente de continua
    integral = cumulative_trapezoid(mensaje_centrado, dx=1 / FS, initial=0.0)
    fase = 2 * np.pi * f_c * t + 2 * np.pi * f_d * integral
    return AMPLITUD_PORTADORA * np.cos(fase)

# Modulador PM
def modular_pm(mensaje, f_c, k_p, t):
    mensaje_centrado = mensaje - np.mean(mensaje)   # Quito la componente de continua
    fase = 2 * np.pi * f_c * t + k_p * mensaje_centrado
    return AMPLITUD_PORTADORA * np.cos(fase)

# Funcion que arma las correspondientes transmisiones en los distintos esquemas de modulacion
def armar_transmisiones(t, mensajes):
    transmisiones = {nombre: {"canales": {}} for nombre in ESQUEMAS}

    # Modula cada mensaje en los distintos canales para cada esquema de modulacion
    for canal, mensaje in mensajes.items():
        portadora = FRECUENCIAS_PORTADORAS_HZ[canal]    # Uso la misma portadora para todos

        transmisiones["FM angosta"]["canales"][canal] = modular_fm(mensaje, portadora, DESVIACION_FM_ANGOSTA_HZ, t)
        transmisiones["FM ancha"]["canales"][canal] = modular_fm(mensaje, portadora, DESVIACION_FM_ANCHA_HZ, t)
        transmisiones["PM"]["canales"][canal] = modular_pm(mensaje, portadora, CONSTANTE_DESVIACION_FASE_PM_RAD, t)
        transmisiones["AM"]["canales"][canal], _ = modular_am(mensaje, portadora, t)

    # Cada esquema transmite sus canales simultaneamente por division de frecuencia.
    for transmision in transmisiones.values():
        transmision["componer"] = np.sum(
            list(transmision["canales"].values()), axis=0  
        )
    return transmisiones