import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly

from configuracion import (
	AMPLITUD_PORTADORA,
	FRACCION_POTENCIA,
	FRECUENCIA_PORTADORA_AM_HZ,
	FRECUENCIA_TONO_HZ,
	FRECUENCIAS_PORTADORAS_HZ,
	FS,
	INDICE_MODULACION_AM,
	OUTPUT_DIR,
)
from modulacion import modular_am


DIRECTORIO_GRAFICOS = OUTPUT_DIR / "graficos"
DURACION_ZOOM_AUDIO_SEG = 0.02
MUESTRAS_POR_PIXEL = 400
DURACION_VISTA_MODULACION_SEG = 0.002
INICIO_VISTA_MODULACION_SEG = 1.5
FACTOR_INTERPOLACION_GRAFICOS = 8


def _guardar(figura, nombre):
	DIRECTORIO_GRAFICOS.mkdir(parents=True, exist_ok=True)
	ruta = DIRECTORIO_GRAFICOS / nombre
	figura.savefig(ruta, dpi=150, bbox_inches="tight")
	plt.close(figura)
	return ruta

# Minimo y maximo por bloque: conserva la envolvente al reducir puntos
def _resumir_por_intervalos(senial, muestras=MUESTRAS_POR_PIXEL):
	cantidad = (len(senial) + muestras - 1) // muestras
	relleno = cantidad * muestras - len(senial)
	bloques = np.pad(senial, (0, relleno), mode="edge").reshape(cantidad, muestras)
	return bloques.min(axis=1), bloques.max(axis=1)


def _espectro(senial):
	"""Espectro unilateral; devuelve frecuencia, magnitud relativa y potencia."""
	n = len(senial)
	ventana = np.hanning(n) if n > 2 else np.ones(n)
	espectro = np.fft.rfft(senial * ventana)
	frecuencias = np.fft.rfftfreq(n, d=1 / FS)
	potencia = np.abs(espectro) ** 2
	if n % 2 == 0:
		potencia[1:-1] *= 2
	else:
		potencia[1:] *= 2
	magnitud_db = 20 * np.log10(
		np.maximum(np.abs(espectro) / max(np.max(np.abs(espectro)), 1e-15), 1e-12)
	)
	return frecuencias, magnitud_db, potencia


def _ancho_mensaje_99(frecuencias, potencia):
	total = potencia.sum()
	if total == 0:
		return 0.0
	indice = np.searchsorted(
		np.cumsum(potencia) / total, FRACCION_POTENCIA
	)
	return float(frecuencias[min(indice, len(frecuencias) - 1)])


def _dibujar_senal_original(eje, t, mensaje, canal, muestras):
	minimos, maximos = _resumir_por_intervalos(mensaje, muestras)
	centros = (np.arange(len(minimos)) + 0.5) * muestras / FS
	eje.fill_between(centros, minimos, maximos, alpha=0.5, linewidth=0)
	eje.set(
		title=f"Señal original completa: {canal} ({t[-1]:.2f} s)",
		xlabel="Tiempo (s)",
		ylabel="Amplitud (min-max por intervalo)",
		xlim=(0, t[-1]),
	)
	eje.grid(True, alpha=0.3)


def _dibujar_senal_demodulada(eje, t, senial):
	if len(senial) > MUESTRAS_POR_PIXEL:
		minimos, maximos = _resumir_por_intervalos(senial)
		centros = t[0] + (np.arange(len(minimos)) + 0.5) * MUESTRAS_POR_PIXEL / FS
		eje.fill_between(centros, minimos, maximos, alpha=0.5, linewidth=0)
	else:
		eje.plot(t, senial, linewidth=0.8)
	eje.grid(True, alpha=0.3)


def generar_graficos(t, mensajes, transmisiones, recuperadas):
	if len(t) < 2 or not mensajes:
		raise ValueError("Se necesitan al menos dos muestras y un mensaje para graficar.")
	if any(len(mensaje) != len(t) for mensaje in mensajes.values()):
		raise ValueError("Los mensajes y el vector de tiempo deben tener igual longitud.")

	esquemas = list(transmisiones.items())
	rutas = []
	mensaje_compuesto = np.sum(list(mensajes.values()), axis=0)
	canales = list(mensajes)

	margen = int(0.02 * FS)
	vista = slice(margen, -margen) if len(t) > 2 * margen else slice(None)
	tiempo_demodulado = t[vista]
	muestras_tono = min(
		len(tiempo_demodulado),
		int(np.floor(4 * FS / FRECUENCIA_TONO_HZ)) + 1,
	)
	figura, ejes = plt.subplots(
		len(esquemas),
		len(canales) + 1,
		figsize=(15, 3 * len(esquemas)),
		sharex="col",
		squeeze=False,
	)
	for fila, (esquema, _) in enumerate(esquemas):
		if esquema not in recuperadas:
			raise ValueError(f"No hay señales demoduladas para el esquema {esquema}.")
		if any(canal not in recuperadas[esquema] for canal in canales):
			raise ValueError(f"Faltan canales demodulados para el esquema {esquema}.")

		seniales = [
			np.sum([recuperadas[esquema][canal] for canal in canales], axis=0),
			*(recuperadas[esquema][canal] for canal in canales),
		]
		for columna, (canal, senial) in enumerate(zip(["Total", *canales], seniales)):
			if len(senial) != len(t):
				raise ValueError(
					f"La señal demodulada {canal} ({esquema}) no coincide con el tiempo."
				)
			eje = ejes[fila, columna]
			if canal == "tono":
				tiempo_tono = tiempo_demodulado[:muestras_tono]
				tiempo_tono = (tiempo_tono - tiempo_tono[0]) * 1000
				eje.plot(
					tiempo_tono,
					senial[vista][:muestras_tono],
					linewidth=0.8,
				)
				eje.set_xlim(0, 4_000 / FRECUENCIA_TONO_HZ)
				eje.grid(True, alpha=0.3)
			else:
				_dibujar_senal_demodulada(eje, tiempo_demodulado, senial[vista])
			eje.set_title(f"{esquema}: {canal}")
			if columna == 0:
				eje.set_ylabel("Amplitud")
	ejes[-1, 0].set_xlabel("Tiempo (s)")
	ejes[-1, 1].set_xlabel("Tiempo (ms)")
	ejes[-1, 2].set_xlabel("Tiempo (s)")
	figura.tight_layout()
	rutas.append(_guardar(figura, "seniales_demoduladas.png"))

	# Se omite la vista completa del tono: es periodico y el detalle basta.
	figura = plt.figure(figsize=(13, 8))
	disposicion = figura.add_gridspec(2, 2)
	eje_tono = figura.add_subplot(disposicion[0, :])
	tono = mensajes.get("tono")
	if tono is not None:
		duracion_tono = min(5 / FRECUENCIA_TONO_HZ, t[-1])
		mostrar_tono = t <= duracion_tono
		eje_tono.plot(t[mostrar_tono] * 1000, tono[mostrar_tono], linewidth=0.8)
		eje_tono.set(
			title=f"Señal original: tono (5 periodos; {duracion_tono * 1000:.1f} ms)",
			xlabel="Tiempo (ms)",
			ylabel="Amplitud",
		)
		eje_tono.grid(True, alpha=0.3)

	if "audio" in mensajes:
		eje_audio = figura.add_subplot(disposicion[1, 0])
		_dibujar_senal_original(
			eje_audio, t, mensajes["audio"], "audio", MUESTRAS_POR_PIXEL
		)
		eje_detalle_audio = figura.add_subplot(disposicion[1, 1])
		inicio = min(0.005, t[-1])
		mostrar_audio = (t >= inicio) & (
			t <= min(inicio + DURACION_ZOOM_AUDIO_SEG, t[-1])
		)
		eje_detalle_audio.plot(
			t[mostrar_audio] * 1000, mensajes["audio"][mostrar_audio], linewidth=0.8
		)
		eje_detalle_audio.set(
			title="Detalle de audio",
			xlabel="Tiempo (ms)",
			ylabel="Amplitud",
		)
		eje_detalle_audio.grid(True, alpha=0.3)
	figura.tight_layout()
	rutas.append(_guardar(figura, "seniales_originales.png"))

	inicio = min(
		INICIO_VISTA_MODULACION_SEG,
		max(0.0, t[-1] - DURACION_VISTA_MODULACION_SEG),
	)
	fin = min(inicio + DURACION_VISTA_MODULACION_SEG, t[-1])
	indice_inicio = int(np.searchsorted(t, inicio))
	indice_fin = int(np.searchsorted(t, fin, side="right"))

	mostrar_compuesto = slice(indice_inicio, indice_fin)
	tiempos_compuestos_ms = (t[mostrar_compuesto] - inicio) * 1000
	figura, eje = plt.subplots(figsize=(13, 5))
	eje.plot(
		tiempos_compuestos_ms,
		mensaje_compuesto[mostrar_compuesto],
		linewidth=0.8,
		label="Audio + tono",
	)
	eje.set(
		title="Señal compuesta de entrada: audio + tono",
		xlabel="Tiempo desde el inicio de la ventana (ms)",
		ylabel="Amplitud",
		xlim=(0, (fin - inicio) * 1000),
	)
	eje.grid(True, alpha=0.3)
	eje.legend(loc="upper right")
	figura.tight_layout()
	rutas.append(_guardar(figura, "senial_compuesta.png"))

	margen = 64
	indice_margen_inicio = max(0, indice_inicio - margen)
	indice_margen_fin = min(len(t), indice_fin + margen)
	tiempo_margen = t[indice_margen_inicio]
	tiempo_interpolado = tiempo_margen + (
		np.arange((indice_margen_fin - indice_margen_inicio) * FACTOR_INTERPOLACION_GRAFICOS)
		/ (FS * FACTOR_INTERPOLACION_GRAFICOS)
	)
	indice_recorte_inicio = (
		indice_inicio - indice_margen_inicio
	) * FACTOR_INTERPOLACION_GRAFICOS
	indice_recorte_fin = (
		indice_fin - indice_margen_inicio
	) * FACTOR_INTERPOLACION_GRAFICOS
	tiempo_transmision_ms = (
		tiempo_interpolado[indice_recorte_inicio:indice_recorte_fin] - inicio
	) * 1000
	figura, ejes = plt.subplots(
		len(esquemas), 1, figsize=(15, 3 * len(esquemas)), squeeze=False
	)
	for eje, (esquema, transmision) in zip(ejes[:, 0], esquemas):
		senial_total = transmision["componer"]
		if esquema == "AM":
			senial_total, _ = modular_am(
				mensaje_compuesto, FRECUENCIA_PORTADORA_AM_HZ, t
			)
			envolvente_total = AMPLITUD_PORTADORA * np.abs(
				1 + INDICE_MODULACION_AM * mensaje_compuesto
			)
			etiqueta_senal = "Transmision total"
			etiquetas_envolvente = (
				"Envolvente +",
				"Envolvente -",
			)
		else:
			etiqueta_senal = "Transmision total"

		senial_interpolada = resample_poly(
			senial_total[indice_margen_inicio:indice_margen_fin],
			FACTOR_INTERPOLACION_GRAFICOS,
			1,
		)
		recorte_senial = senial_interpolada[
			indice_recorte_inicio:indice_recorte_fin
		]
		eje.plot(
			tiempo_transmision_ms,
			recorte_senial,
			linewidth=0.8,
			label=etiqueta_senal,
		)
		if esquema == "AM":
			envolvente_interpolada = resample_poly(
				envolvente_total[indice_margen_inicio:indice_margen_fin],
				FACTOR_INTERPOLACION_GRAFICOS,
				1,
			)
			recorte_envolvente = envolvente_interpolada[
				indice_recorte_inicio:indice_recorte_fin
			]
			eje.plot(
				tiempo_transmision_ms,
				recorte_envolvente,
				"--",
				linewidth=0.9,
				label=etiquetas_envolvente[0],
			)
			eje.plot(
				tiempo_transmision_ms,
				-recorte_envolvente,
				"--",
				linewidth=0.9,
				label=etiquetas_envolvente[1],
			)
		eje.set(
			title=f"{esquema}: señal transmitida",
			ylabel="Amplitud",
			xlim=(0, max(tiempo_transmision_ms[-1], DURACION_VISTA_MODULACION_SEG * 1000)),
		)
		eje.grid(True, alpha=0.3)
		eje.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
	figura.tight_layout(rect=(0, 0, 0.76, 1))
	rutas.append(_guardar(figura, "seniales_moduladas_total.png"))

	canales = list(mensajes)
	figura, ejes = plt.subplots(
		len(esquemas),
		len(canales),
		figsize=(7 * len(canales), 3 * len(esquemas)),
		squeeze=False,
	)
	for fila, (esquema, transmision) in enumerate(esquemas):
		for columna, canal in enumerate(canales):
			senial_canal = transmision["canales"][canal]
			if len(senial_canal) != len(t):
				raise ValueError(
					f"La señal modulada del canal {canal} ({esquema}) no coincide con el tiempo."
				)

			if esquema == "AM":
				mensaje_centrado = mensajes[canal] - np.mean(mensajes[canal])
				envolvente_canal = AMPLITUD_PORTADORA * np.abs(
					1 + INDICE_MODULACION_AM * mensaje_centrado
				)
			else:
				envolvente_canal = np.full_like(t, AMPLITUD_PORTADORA)

			senial_interpolada = resample_poly(
				senial_canal[indice_margen_inicio:indice_margen_fin],
				FACTOR_INTERPOLACION_GRAFICOS,
				1,
			)
			envolvente_interpolada = resample_poly(
				envolvente_canal[indice_margen_inicio:indice_margen_fin],
				FACTOR_INTERPOLACION_GRAFICOS,
				1,
			)
			recorte_senial = senial_interpolada[
				indice_recorte_inicio:indice_recorte_fin
			]
			recorte_envolvente = envolvente_interpolada[
				indice_recorte_inicio:indice_recorte_fin
			]
			eje = ejes[fila, columna]
			eje.plot(tiempo_transmision_ms, recorte_senial, linewidth=0.8, label="Señal modulada")
			eje.plot(
				tiempo_transmision_ms,
				recorte_envolvente,
				"--",
				linewidth=0.9,
				label="Envolvente +",
			)
			eje.plot(
				tiempo_transmision_ms,
				-recorte_envolvente,
				"--",
				linewidth=0.9,
				label="Envolvente -",
			)
			eje.set(
				title=f"{esquema}: canal {canal}",
				xlabel="Tiempo desde el inicio de la ventana (ms)",
				ylabel="Amplitud",
				xlim=(0, max(tiempo_transmision_ms[-1], DURACION_VISTA_MODULACION_SEG * 1000)),
			)
			eje.grid(True, alpha=0.3)
			eje.legend(loc="upper left")
	figura.tight_layout()
	rutas.append(_guardar(figura, "seniales_moduladas_por_canal.png"))

	espectros_mensajes = {
		canal: _espectro(mensaje) for canal, mensaje in mensajes.items()
	}
	anchos_mensajes = {
		canal: _ancho_mensaje_99(espectro[0], espectro[2])
		for canal, espectro in espectros_mensajes.items()
	}
	for canal, mensaje in mensajes.items():
		# Espectro completo de baseband y bandas de cada portadora por separado.
		f_mensaje, db_mensaje, _ = espectros_mensajes[canal]
		ancho_mensaje = anchos_mensajes[canal]
		figura, ejes = plt.subplots(
			len(esquemas) + 1, 1, figsize=(12, 2.8 * (len(esquemas) + 1))
		)
		ejes[0].plot(f_mensaje, db_mensaje, linewidth=0.8)
		ejes[0].set(
			title=f"Mensaje",
			xlabel="Frecuencia (Hz)",
			ylabel="Magnitud (dB)",
			xlim=(0, min(FS / 2, max(3 * ancho_mensaje, 2_500))),
			ylim=(-100, 3),
		)
		ejes[0].grid(True, alpha=0.3)

		for eje, (esquema, transmision) in zip(ejes[1:], esquemas):
			portadora = FRECUENCIAS_PORTADORAS_HZ[canal]
			senial_modulada = transmision["canales"][canal]
			frecuencias, magnitud, _ = _espectro(senial_modulada)

			eje.plot(frecuencias, magnitud, linewidth=0.8)
			eje.set(
				title=f"{esquema}: {canal}",
				xlabel="Frecuencia (Hz)",
				ylabel="Magnitud (dBc)",
				xlim=(
					max(0, portadora - 2_000),
					min(FS / 2, portadora + 2_000),
				),
				ylim=(-100, 3),
			)
			eje.grid(True, alpha=0.3)

		figura.tight_layout()
		rutas.append(_guardar(figura, f"espectros_{canal}.png"))

	# Muestra el espectro de cada transmisión FDM.
	figura, ejes = plt.subplots(
		len(esquemas), 1, figsize=(12, 3 * len(esquemas)), squeeze=False
	)
	for eje, (esquema, transmision) in zip(ejes[:, 0], esquemas):
		frecuencias, magnitud, _ = _espectro(transmision["componer"])
		eje.plot(frecuencias, magnitud, linewidth=0.8, label="Señal compuesta")
		eje.set(
			title=f"Espectro compuesto: {esquema}",
			xlabel="Frecuencia (Hz)",
			ylabel="Magnitud (dBc)",
			xlim=(0, FS / 2),
			ylim=(-100, 3),
		)
		eje.grid(True, alpha=0.3)
		eje.legend(loc="upper right")

	figura.tight_layout()
	rutas.append(_guardar(figura, "espectros_compuestos.png"))
	return rutas