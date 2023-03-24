import numpy as np
import matplotlib.pyplot as plt

# Crea un eje de tiempo
t = np.linspace(0, 2*np.pi, 500)

# Crea dos ondas
wave1 = np.sin(t)
wave2 = np.sin(5*t)

# Superpone las dos ondas
resultant = wave1 + wave2

# Crea el plot
plt.plot(t, wave1, label='Onda 1')
plt.plot(t, wave2, label='Onda 2')
plt.plot(t, resultant, label='Resultante', linewidth=2)
plt.legend()
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Superposición de dos ondas')
plt.show()
