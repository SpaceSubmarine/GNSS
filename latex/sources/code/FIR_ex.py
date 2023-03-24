import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Crear una señal de prueba que incrementa su frecuencia con el tiempo
fs = 1000 # Frecuencia de muestreo
t = np.arange(0, 1, 1/fs)
f1 = 10 # Frecuencia inicial
f2 = 200 # Frecuencia final
x = np.sin(2*np.pi*f1*t + 2*np.pi*(f2-f1)*t**2)

# Diseñar el filtro de paso bajo FIR
fc = 50 # Frecuencia de corte del filtro
numtaps = 101 # Número de coeficientes del filtro
h = signal.firwin(numtaps, fc/(fs/2), window='hamming')

# Aplicar el filtro a la señal
y = signal.convolve(x, h, mode='same')

# Graficar la señal original y la señal suavizada
plt.plot(t, x, label='Señal original')
plt.plot(t, y, label='Señal suavizada')
plt.legend()
plt.show()