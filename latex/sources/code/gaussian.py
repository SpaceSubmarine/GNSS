import numpy as np


D0 = 0.001  # m
#Esta fórmula describe la relación entre el radio (W) de un haz láser Gaussiano y su distancia axial (z) desde la posición del radio (z0). W0 representa el radio mínimo en z0 y zR es el rango de Rayleigh, que es una medida de la difracción del haz. La fórmula indica que el radio (W) se calcula multiplicando el radio mínimo (W0) por la raíz cuadrada de 1 más el cuadrado de la relación entre la distancia (z - z0) y el rango de Rayleigh (zR).

"""
Parameters:
z (float): The axial distance from the waist position.
W0 (float): The minimum waist size at the waist position.
z0 (float): The position of the waist.
zR (float): The Rayleigh range, which is a measure of the beam's diffraction.
"""
def waist_size(z, W0, z0, zR):
    return W0 * (1 + ((z - z0) / zR) ** 2) ** 0.5

lenght = 3000  #m
lenght = lenght * 100  #cm

lenght = np.linspace(0, lenght, lenght+1)
radius = np.zeros_like(lenght)

for i in lenght:
    radius[i] = waist_size()  # Pendiente introducir inputs y comprobar


