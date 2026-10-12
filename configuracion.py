from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_AUDIO = ROOT/"media"/"faint.wav" 
OUTPUT_DIR = ROOT/"resultados" 

FS = 192_000                        # Frecuencia de muestreo en Hz.
FRECUENCIA_CORTE_AUDIO_HZ = 7_000   # Frecuencia de corte del audio 
FRECUENCIA_TONO_HZ = 1_000          # Frecuencia del tono
AMPLITUD_TONO = 0.5                 # Amplitud del tono

FRECUENCIAS_PORTADORAS_HZ = {"tono": 25_000, "audio": 65_000}
FRECUENCIA_PORTADORA_AM_HZ = FRECUENCIAS_PORTADORAS_HZ["audio"]
AMPLITUD_PORTADORA = 1.0             # Amplitud de las portadoras

INDICE_MODULACION_AM = 0.8           # Indice de modulación AM
DESVIACION_FM_ANGOSTA_HZ = 500      # Desviación FM en banda angosta
DESVIACION_FM_ANCHA_HZ = 12_000     # Desviación FM en banda ancha
CONSTANTE_DESVIACION_FASE_PM_RAD = 2.5  # Constante de desviación de Fase en PM
GUARDA_HZ = 1_000                   # Guarda entre los canales 
ORDEN_FILTRO = 6                    # Orden del filtro
FRACCION_POTENCIA = 0.99            # Fracción de la potencia total de la señal que cae dentro de la banda ocupada por el canal

ESQUEMAS = ("AM", "FM angosta", "FM ancha", "PM")
