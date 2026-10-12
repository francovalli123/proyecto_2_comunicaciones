# Programa principal

import numpy as np
import soundfile as sf

from configuracion import (
	AMPLITUD_TONO,
	FRECUENCIA_TONO_HZ,
	FS,
	OUTPUT_DIR,
)
from demodulacion import recuperar_seniales
from modulacion import armar_transmisiones
from audio import cargar_audio
from graficos import generar_graficos
from analisis import analizar_rendimiento

# Guarda las señales demoduladas, separadas por esquema y canal
def guardar_seniales_demoduladas(recuperadas):
	directorio_salida = OUTPUT_DIR
	directorio_salida.mkdir(parents=True, exist_ok=True)
	rutas = {}

	for esquema, canales in recuperadas.items():
		nombre_esquema = esquema.replace(" ", "_")	# Reemplaza los espacios con _ (para que crear las carpetas por esquema)
		carpeta_esquema = directorio_salida / nombre_esquema 
		carpeta_esquema.mkdir(parents=True, exist_ok=True)	# Crea la carpeta del esquema
		rutas[esquema] = {}

		for canal, senial in canales.items():
			ruta = carpeta_esquema / f"{canal}.wav"	# Por ejemplo, AM/audio.wav
			# WAV FLOAT conserva el rango de la señal recuperada sin normalizarla
			sf.write(ruta, senial, FS, format="WAV", subtype="FLOAT")
			rutas[esquema][canal] = ruta

	return rutas

# Prepara los mensajes, ejecuta modulacion y demodulacion y devuelve los resultados
def ejecutar():
	audio, audio_referencia, tiempo = cargar_audio()
	tono = AMPLITUD_TONO * np.sin(2 * np.pi * FRECUENCIA_TONO_HZ * tiempo)
	mensajes = {"tono": tono, "audio": audio}

	transmisiones = armar_transmisiones(tiempo, mensajes)
	recuperadas = recuperar_seniales(transmisiones, mensajes, tiempo)
	archivos_salida = guardar_seniales_demoduladas(recuperadas)
	archivos_graficos = generar_graficos(tiempo, mensajes, transmisiones, recuperadas)
	analizar_rendimiento(mensajes, transmisiones, recuperadas)

	return {
		"recuperadas": recuperadas,
		"archivos_salida": archivos_salida,
		"archivos_graficos": archivos_graficos,
	}

def main():
	resultados = ejecutar()
	print(f"Archivos WAV demodulados guardados en: {OUTPUT_DIR / 'demoduladas'}")
	print(f"Graficas guardadas en: {OUTPUT_DIR / 'graficos'}")
	return resultados

if __name__ == "__main__":
	main()
