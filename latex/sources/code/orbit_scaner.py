import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import math

H = 693 * 10 ** 3  # Satellite height
betta = 90  # degrees of visión
thick_factor = 1
G = 6.67430 * (10 ** (-11))  # [N·m2]/[kg2]
M = 5.9722 * (10 ** 24)  # [kg]
# cálculo de la velocidad orbital de un satelite en orbita circular

def velocidad_orbital(altura):
    G = 6.67430 * (10 ** (-11))  # [N·m2]/[kg2]
    R = 6371 * 1000  # [m] asumiendo tierra 100% esferica
    h_total = altura + R  # [m]
    M = 5.9722 * (10 ** 24)  # [kg]
    V_orbital = math.sqrt((G * M) / h_total)  # [m/s]
    return V_orbital, h_total


def base_isosceles(angulo1, angulo2, angulo3, altura):
    # Verificamos que los ángulos sumen 180 grados
    if angulo1 + angulo2 + angulo3 != 180:
        return "Los ángulos no suman 180 grados, no es un triángulo válido"

    # Verificamos que sea un triángulo isósceles (2 ángulos iguales)
    if angulo1 != angulo2 and angulo2 != angulo3 and angulo1 != angulo3:
        return "No es un triángulo isósceles"

    # Calculamos el ángulo que no es igual a los demás
    if angulo1 == angulo2:
        angulo = angulo3
    elif angulo2 == angulo3:
        angulo = angulo1
    else:
        angulo = angulo2

    # Calculamos la base utilizando la fórmula
    base = 2 * altura * math.tan(math.radians(angulo / 2))

    return base


ang_1 = (180 - betta) / 2
ang_2 = ang_1
d_base = base_isosceles(betta, ang_1, ang_2, H)
v_orb, h_total = velocidad_orbital(H)
time = d_base / v_orb
area = d_base*(d_base*thick_factor)
t_orb = 2 * np.pi * math.sqrt((h_total**3)/(G*M))

print("Período orbital:", round(t_orb/60, 3), "[minutos]")
print("Altura del satélite:", round(H / 1000, 3), "[km]")
print("Ángulo de visión:", round(betta, 3), "º, ", round((180-betta)/2, 3), "º respecto al horizonte")
#print("Amplitud entre muestras de ambos barridos:", round(d_base / 1000, 3), "[km]")
print("Tiempo de muestra:", round(time/60, 3), "[minutos]")
print("Velocidad orbital", round(v_orb, 3), "[m/s]")
print("Area fotográfica:", round(area/1000/1000/1000/1000, 3), "·10⁶[km2]")
print("Rango del campo de visión de extremo a extremo:", round(d_base/1000, 3), "[km]")
print("Cantidad de áreas por órbita:", round(t_orb/time, 3))





'''set up orthographic map projection with
perspective of satellite looking down at 50N, 100W.
use low resolution coastlines.
don't plot features that are smaller than 1000 square km.'''
map = Basemap(projection='ortho', lat_0 = 50, lon_0 = -100,
resolution = 'l', area_thresh = 1000.)

'''draw coastlines, country boundaries, fill continents.'''

map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color = 'coral')
'''draw the edge of the map projection region (the projection limb)'''
map.drawmapboundary()

'''draw lat/lon grid lines every 30 degrees.'''
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))

'''Add representation of Sentinel 1 satellite'''
x, y = map(-100, 50)
plt.plot(x, y, marker='o', markersize=5, color='red')
plt.text(x, y, 'Sentinel 1', fontsize=10, color='red',
ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.5))

plt.show()
