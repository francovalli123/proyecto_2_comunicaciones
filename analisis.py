import numpy as np

def _potencia(senal):
	# Potencia media de una señal discreta.
	return np.mean(senal ** 2)

def _snr(referencia, recuperada):
	potencia_senal = _potencia(referencia)
	potencia_ruido = _potencia(referencia - recuperada) # Interpreto la diferencia entre ambas señales como ruido o error
	if potencia_senal == 0:
		return None
	return np.inf if potencia_ruido == 0 else 10 * np.log10(potencia_senal / potencia_ruido)

# Calcula e imprime la potencia de señal y la SNR de cada esquema
def analizar_rendimiento(mensajes, transmisiones, recuperadas):
	mensaje = np.sum(list(mensajes.values()), axis=0)

	for esquema, transmision in transmisiones.items():
		mensaje_recuperado = np.sum(list(recuperadas[esquema].values()), axis=0)
		print(esquema)
		print(f"potencia mensaje = {_potencia(mensaje):.6g}")
		print(f"potencia total = {_potencia(transmision['componer']):.6g}")
		print(f"relacion S/N = {_snr(mensaje, mensaje_recuperado)} dB\n")
