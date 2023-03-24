import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import time

start = time.time()
E = 5e-3  # Pulse energy (J)
tau = 500e-9  # Pulse duration (s)
BN = 128e6  # Noise bandwidth (Hz)
eta = 0.1  # Photodetector quantum efficiency
beta = 1e-7  # Backscatter coefficient (m-1 sr-1)
S = 50  # Lidar ratio (sr)
Lambda = 1.064e-6  # Wavelength (m)
D = 0.1  # Aperture diameter (m)
h = 6.6e-34  # Planck's constant (J s)
Cn2 = 1e-14  # Structure constant of index of refraction (m-2/3)

for n in range(3, 7):
    # Plots SNR for different curvature radii 10^n in m of the output beam
    F = 10 ** n

    # Structure constant of index of refraction (m-2/3)
    Cn2 = 1e-14

    R = np.array(range(1, 5001))

    SNRtau = (E * eta * tau * beta * Lambda * np.exp(-2 * S * beta * R) * np.pi * D ** 2) / \
             (8 * h * R ** 2 * (1 + (1 - R / F) ** 2 * (np.pi * D ** 2 / (4 * Lambda * R)) ** 2 +
                                (D / (2 * 0.069 * Lambda ** (6 / 5) * (Cn2 * R) ** (-3 / 5))) ** 2))

    SNRBN = (E * eta * beta * Lambda * np.exp(-2 * S * beta * R) * np.pi * D ** 2) / \
            (8 * BN * h * R ** 2 * (1 + (1 - R / F) ** 2 * (np.pi * D ** 2 / (4 * Lambda * R)) ** 2 +
                                    (D / (2 * 0.069 * Lambda ** (6 / 5) * (Cn2 * R) ** (-3 / 5))) ** 2))

    # plt.style.use("dark_background")

    plt.figure(1)
    plt.grid(color="grey", lw=0.3)
    # plt.xscale('log')
    plt.title("Signal-to-noise ratio SNR_tau as a function of resolution bandwidth")
    plt.xlabel('Bn')
    plt.ylabel('10log(SNR)')
    if n == 3:
        plt.plot(R, 10 * np.log10(SNRtau), c="k", lw=0.5, label="n=3")
    elif n == 4:
        plt.plot(R, 10 * np.log10(SNRtau), c="r", lw=0.5, label="n=4")
    elif n == 5:
        plt.plot(R, 10 * np.log10(SNRtau), c="b", lw=0.5, label="n=5")
    elif n == 6:
        plt.plot(R, 10 * np.log10(SNRtau), c="g", lw=0.5, label="n=6")
    plt.legend()

    plt.figure(2)
    plt.grid(color="grey", lw=0.3)
    plt.title("Signal-to-noise ratio SNR_BN as a function of resolution bandwidth")
    plt.xlabel('Bn')
    plt.ylabel('10log(SNR)')
    if n == 3:
        plt.plot(R, 10 * np.log10(SNRBN), c="k", lw=0.5, label="n=3")
    elif n == 4:
        plt.plot(R, 10 * np.log10(SNRBN), c="r", lw=0.5, label="n=4")
    elif n == 5:
        plt.plot(R, 10 * np.log10(SNRBN), c="b", lw=0.5, label="n=5")
    elif n == 6:
        plt.plot(R, 10 * np.log10(SNRBN), c="g", lw=0.5, label="n=6")
    plt.legend()
plt.show()

# DATAFRAME
end = time.time()
print("Building table of results...")

data = [
    ["The back-scatter coefficient $\\beta$", beta, "$m^{-1}sr^{-1}$"],
    ["Pulse energy $E$", E, "$J$"],
    ["The duration of the pulse $\\tau$", tau * 10 ** 9, "$ns$"],
    ["The noise bandwidth $B_{N}$", BN * 10 ** (-6), "$MHz$"],
    ["The quantum efficiency of the photodiode detector $\eta$", eta, ""],
    ["The lidar ratio $S$", S, "$sr$"],
    ["The wavelength $\\lambda$", Lambda * 10 ** 6, "$m$"],
    ["The opening diameter $D$", D, "$m$"],
    ["Planck's constant $h$", h, "$Js$"],
    ["The structure constant of the refractive index $Cn^2$", Cn2, "$m^{-2/3}$"],
    ["Computational time ", end - start, '$s$']
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
print("Values used for calculation:\\\\")
print("")
print("\\begin{table}[h]")
print("\\centering")
print("\\begin{center}")
print(latex_table)
print("\\end{center}")
print("\\caption{SNR Simulation Values}")
print("\\end{table}")
print("")
