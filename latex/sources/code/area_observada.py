import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from tabulate import tabulate
import time
start = time.time()

# https://www.foto24.com/super-teleobjetivo-de-espejo-samyang-500mm-f-6-3-canon.html

F = 0.5  # Distancia focal del objetivo (m)
L = 1000  # Distancia del area de observacion
N = 5000  # round((F * 10) + 1)
alpha = 5  # angulo de vision del objetivo 500mm
D = 2 * F * math.tan(math.radians(alpha/2))  # diametro del objetivo en (m)


def area_visual(diametro, focal, distancia):
    alpha = 2 * math.degrees(math.atan(diametro / (2 * focal)))
    A = (distancia ** 2) * alpha

    return alpha, A


L_range = np.linspace(0, L, N)  # discretizacion de la distancia L en N elementos
F_range = np.linspace(0, F, N)  # discretizacion de la distancia focal F en N elementos
alpha_range = np.zeros_like(L_range)  # inicializacion del angulo en N elementos
A_range = np.zeros_like(L_range)  # inicializacion del Area de visualización para L_range
alpha_range_2 = np.zeros_like(L_range)  # inicializacion del angulo en N elementos
A_range_2 = np.zeros_like(L_range)  # inicializacion del Area de visualización para L_range

# Calculo del angulo de visión y Area de visión para todas las distancias de vision
for i in range(len(L_range)):
    alpha_range[i], A_range[i] = area_visual(D, F, L_range[i])
# Calculo del angulo de visión y Area de visión para todas las distancias focales
for i in range(len(F_range)):
    alpha_range_2[i], A_range_2[i] = area_visual(D, F_range[i], L)

# PLOT SECTION
# plt.style.use("dark_background")




print("Viewed area (virtual image):", np.max(A_range)/1000/1000, "km2")


# DATAFRAME
end = time.time()
print("Building table of results...")
data = [
    ["Telephoto diameter", D, "$m$"],
    ["Image distance", L, "$m$"],
    ["Field of view angle $\\alpha$", alpha, "$º$"],
    ["Focal distance $F$", F, "$m$"],
    ["Viewed area (virtual image):", np.max(A_range)/1000/1000, "$km^2$"],
    ["Computational time ", end-start, '$s$']
    ]


df = pd.DataFrame(data, columns=['Description', 'Variable', 'Units'])
latex_table = tabulate(df, headers='keys', tablefmt='latex', floatfmt='.2E')
latex_table = latex_table.replace('\$m\^{}\{-1\}sr\^{}\{-1\}\$', '$m^{-1}sr^{-1}$')
latex_table = latex_table.replace('\$beta\$', '$\\beta$')
latex_table = latex_table.replace('\$', '$')
latex_table = latex_table.replace('textbackslash{}', '')
latex_table = latex_table.replace('\}', '}')
latex_table = latex_table.replace('\_', '_')
latex_table = latex_table.replace('\{', '{')
latex_table = latex_table.replace('\^{}', '^')
print("")
print("Values used for Telephoto: Samyang 500mm f/6.3 Canon Super Telephoto Mirror Lens:\\\\")
print("")
print("\\begin{center}")
print(latex_table)
print("\\end{center}")
print("")


fig, axs = plt.subplots(1, 2, figsize=(11, 4))

axs[0].stackplot(L_range, A_range / 1000 / 1000, colors=["RebeccaPurple"])
axs[0].autoscale()
axs[0].set_title("Variation of area by observed distance, diameter: " + str("{:.2f}".format(D*1000)) + "(mm)")
axs[0].set_xlabel('Object distance in meters')
axs[0].set_ylabel('Area in km2')
axs[0].grid(color="grey")

axs[1].stackplot(F_range, A_range_2 / 1000 / 1000, colors=["DarkCyan"])
axs[1].autoscale()
axs[1].set_title("Variation of area per focal length a" + str(L) + "(m)")
axs[1].set_xlabel('Focal distance in meters')
axs[1].set_ylabel('Area in km2')
axs[1].grid(color="grey")
plt.autoscale()
plt.show()
