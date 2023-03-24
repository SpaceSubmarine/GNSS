import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import numpy as np
import torch
import time
start = time.time()

# INPUT SECTION
black_ground = False
f1 = 500e6  # First frequency (Hz)
f2 = 1000e6  # Second frequency (Hz)
f3 = 500e6  # Third frequency (Hz)
delta_t = 1e-11  # resolution time
amplitude = 1
fig_path = "Frame_generation/MatPlot/LASER/"
n1 = 100
n2 = 400
n3 = 100


# CALC
print("Doing calculations...")
t1 = (1 / f1)  # Period of f1
t2 = (1 / f2)  # Period of f2
t3 = (1 / f3)  # Period of f2
# delta_t = t3 / resolution
t_wave = (n1 * t1 + n2 * t2 + n3 * t3)
tw1 = torch.linspace(0, t1, int(t1/delta_t))
tw2 = torch.linspace(0, t2, int(t2/delta_t))
tw3 = torch.linspace(0, t3, int(t3/delta_t))
omega_1 = 2 * np.pi * f1 * tw1
omega_2 = 2 * np.pi * f2 * tw2
omega_3 = 2 * np.pi * f3 * tw3
A1 = torch.sin(omega_1)
A2 = torch.sin(omega_2)
A3 = torch.sin(omega_3)
A4 = torch.cat((torch.tile(A1, (n1,)), torch.tile(A2, (n2,)), torch.tile(A3, (n3,))))
t = torch.linspace(0, t_wave, len(A4))

print("Calculations DONE!")


# Fast Fourier Transform
print("Doing FFT...")
fft_A4 = torch.fft.fft(A4)
freqs = torch.fft.fftfreq(len(A4), d=t[1]-t[0])
amplitud = 2.0 / (len(t)) * torch.abs(fft_A4)  # Calcula la amplitud de la transformada de Fourier
pos_mask = freqs > 0  # Selecciona solo las frecuencias positivas
neg_mask = freqs < 0  # Selecciona solo las frecuencias negativas
print("FFT DONE!")

# DATAFRAME
print("Building table of results...")
print("")
end = time.time()
data = [['Number of time-steps', len(freqs), ''],
        ['First Frequency', f1 * 10 ** (-3), 'kHZ'],
        ['Second Frequency', f2 * 10 ** (-3), 'kHZ'],
        ['Third Frequency', f3 * 10 ** (-3), 'kHZ'],
        ['Time first Frequency', t1 * 10 ** 9, 'ns'],
        ['Time second Frequency', t2 * 10 ** 9, 'ns'],
        ['Time third Frequency', t3 * 10 ** 9, 'ns'],
        ['Nº first Frequency', n1, ''],
        ['Nº second Frequency', n2, ''],
        ['Nº third Frequency', n3, ''],
        ['Total wave period', t_wave * 10 ** 9, 'ns'],
        ["Computational time ", end-start, 's']
        ]

df = pd.DataFrame(data, columns=['Description', 'Variable', 'Units'])


# Convierte el dataframe a formato LaTeX
latex_table = tabulate(df, headers='keys', tablefmt='latex', floatfmt='.2E')
print("\\begin{table}[h]")
print("\\centering")
print("\\begin{center}")
print(latex_table)
print("\\end{center}")
print("\\caption{FFT Verification Simulation Values}")
print("\\end{table}")
print("")
print("Table of results DONE!")




# PLOT SECTION
print("Plotting...")
if black_ground:
    plt.style.use("dark_background")


# Fast Fourier Transform Plots in dBs
dBs = []
for i in range(len(fft_A4)):
    dBs.append(20 * torch.log(torch.abs(fft_A4[i])))

plt.figure()
plt.plot(freqs[pos_mask], amplitud[pos_mask], c="green", marker='.', label='Positiva')  # Grafica la parte positiva
plt.plot(freqs[neg_mask], amplitud[neg_mask], c="red", marker='.', label='Negativa')  # Grafica la parte negativa
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('amplitude')
plt.title('Transformada de Fourier de A3')
plt.legend()
plt.savefig(fig_path + "test_signal.png")
plt.show()

plt.plot(t, A4, c="green", label='A4')
plt.title('Signals A1 and A2')
plt.xlabel('Time (s)')
plt.ylabel('amplitude')
plt.legend()
plt.show()