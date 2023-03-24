from scipy.signal import chirp
import matplotlib.pyplot as plt
import scipy.signal as signal
from tabulate import tabulate
import pandas as pd
import numpy as np
import math
import time
import os

start = time.time()

# INPUT SECTION
print("Introduce the object distance in meters: ")
d_obj = float(input())  # objet detection at a half distance in meters

print("Make plots? (0 for NO, 1 for YES)")
make_plots = int(input())

if make_plots == 1:
    print("Plots in dark theme? (0 for NO, 1 for YES)")
    black_ground = int(input())

    if black_ground == 1:
        valor_booleano = True
    elif black_ground == 0:
        valor_booleano = False

    if black_ground:
        plt.style.use("dark_background")
        print("Dark Theme ENABLED")
    else:
        print("Dark Theme DISABLED")
    print("Show Plots? (0 for NO, 1 for YES)")
    show_plot = int(input())
    if show_plot == "1":
        show_plot = True
    elif show_plot == "0":
        show_plot = False
    print("Save Plots? (0 for NO, 1 for YES)")
    save_plots = int(input())
    if save_plots == "1":
        save_plots = True
    elif save_plots == "0":
        save_plots = False

fs = 1 * 10 ** 9  # GHz
f0 = 0.9 * 10 ** 9  # GHz
f1 = 1 * 10 ** 9  # GHz
phi_0 = 270  # initial phi for amplitude 0+ in t=0
amplitude = 1  # arbitrary.append(f_actual)
distance = 2000  # tracking distance in meters
light = 299792458  # m/s
go_back = distance * 2  # meters
duration = go_back / light  # seconds
v_clock = 3.5  # GHz
n_clock = 24  # Number of clocks
f_res = v_clock * (10 ** 9) / n_clock  # Frequency resolution in Hz
t_res = (1 / f_res)  # Time resolution in seconds
d_res = light * t_res  # Distance resolution in meters
resolution = 1000  # Number of points per time-step
TOF = (d_obj / light) * 2  # time of distance in seconds
d_phase_shift = (t_res / 2) * light  # time to phase shift
#########################################################################################
TOF2 = t_res * 1000  # time to have 100 times relationship with t_res
dist2 = TOF2 * light  # the distance for TOF2
#########################################################################################
label_A = "Emitted chirp"
label_A2 = "Wave comparison"
label_echo = "Received Chirp"
fig_path = "Frame_generation/MatPlot/LASER/FFT/" + str(int(d_obj)) + "/"
if os.path.exists(fig_path):
    print("Working on a existing folder...")
else:
    print("Creating a new simulation folder...")
    os.mkdir(fig_path)

# FIR
fc = 10 * 10 ** 6  # cut-off frequency
FIR = "FIR, fc:" + str(fc * 10 ** (-6)) + "MHz"
num_taps = 1000  # Number of coefficients


def square_cosine(x):
    return 0.5 * (math.cos(2 * x) + 1)


print("Doing calculations...")
# Frequency steps:
n_step = duration / t_res  # number of increasing frequency steps
delta_f = (f1 - f0) / n_step  # increment of frequency per time-step
peaks = (t_res / (1 / f0))  # number of peaks for the first time-step

# INITIALIZATIONS
t = [0]
frequencies = [f0]  # initial frequency
tactual = [0]
A = [np.sin(amplitude * 2 * np.pi * frequencies[0] * t[0])]  # amplitude along frequency
A2 = [np.sin(amplitude * 2 * np.pi * frequencies[0] * t[0])]  # second wave initialization
t_step = t_res
step_actual = 0
n_step = duration / t_step

for i in range(0, round(n_step)):
    f_actual = f0 + (i * delta_f)
    for n in range(resolution):
        frequencies.append(f_actual)

pi = np.pi
for i in range(len(frequencies) - 1):
    # for n in range(1, peaks):
    step_actual += 1
    t_actual = (t_step / resolution) * step_actual
    t.append(t_actual)
    A.append(chirp(t_actual, f0=f0, f1=f1, t1=duration, method='lin', phi=phi_0))

# ECHO
n = round((TOF / t_res) * resolution)
echo = np.zeros(len(t) + n)  # same size like emitted wave
# Displacement of the values or the distance in time
echo[n:] = A[:len(echo) - n]
A = np.pad(A, (0, n), 'constant')

for i in range(n):
    # for n in range(1, peaks):
    step_actual += 1
    t_actual = (t_step / resolution) * step_actual
    t.append(t_actual)

for i in range(len(t) - 1):
    A2.append((A[i]) * echo[i])

frequencies = []  # frequency reboot
f_actual = f0
count = 0
fcount = 1
for i in range(len(t)):
    if count > resolution:
        f_actual = f0 + (fcount * delta_f)
        count = 0
        fcount += 1
    count += 1
    frequencies.append(f_actual)

# FIR FILTER
h = signal.firwin(num_taps, fc / (fs / 2), window='hamming')
# applying filter
y = signal.convolve(A2, h, mode='same')
print("Calculations DONE!")

# Fast Fourier Transform
print("Doing FFT...")
fft_A4 = np.fft.fft(A2)
freqs = np.fft.fftfreq(len(A2), d=t[1] - t[0])
amplitud = 2.0 / (len(t)) * abs(fft_A4)  # compute the amplitude
pos_mask = freqs > 0  # positive filter
neg_mask = freqs < 0  # negative filter

max_value = np.max(amplitud)  # encuentra el valor máximo en el array
max_index = np.argmax(amplitud)  # encuentra la posición del valor máximo en el array
print("FFT Freq", freqs[max_index])
# print("Relation:", d_obj/freqs[max_index])


print("FFT DONE!")

if make_plots:
    # PLOT SECTION
    print("Plotting...")

    # GENERAL PLOT OF FREQUENCIES N1
    plt.figure()
    plt.plot(t, A, c="SteelBlue", alpha=0.6, marker='.', label=label_A)
    if black_ground == 1:
        plt.plot(t, A, c="white", alpha=0.2, marker='.')
    plt.title("Linear Chirp, f(0)=" + str(f0 * 10 ** (-9)) + "GHz, f(1)=" + str(f1 * 10 ** (-9)) + "GHz for " + str(
        d_obj) + " m ")
    plt.xlabel('t (s), linear scale')
    plt.xscale('linear')
    plt.grid(alpha=0.25, which='both')
    plt.legend()
    plt.xlim([-0.1e-8, 1.1e-8])
    plt.ylim([-1.01, 1.01])
    if save_plots:
        plt.savefig(fig_path + "chirp_Lineal-0-" + str(d_obj) + ".png")
    if show_plot:
        plt.show()
    plt.close()

    # Fast Fourier Transform Plots in dBs
    dBs = []
    for i in range(len(amplitud)):
        dBs.append(20 * np.log(np.abs(amplitud[i])))
    dBs = np.array(dBs)

    plt.figure()
    plt.plot(freqs[pos_mask], dBs[pos_mask], c="SteelBlue", label='Positive')  # plot positive part
    plt.plot(freqs[neg_mask], dBs[neg_mask], c="DarkRed", label='Negative')  # plot negative part
    plt.grid(alpha=0.25, which='both')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel("Amplitude in dBs")
    if d_obj == 1000:
        plt.xlim([-2.2e9, 2.2e9])
    plt.ylim(np.min(dBs) * 0.8, np.max(dBs) * 0.5)
    plt.title('Fast Fourier Transform of the Beat Frequencies for ' + str(d_obj) + " m " + '{:.3f}'.format(
        freqs[max_index] * 10 ** (-6)) + " MHz")
    plt.legend()
    if save_plots:
        plt.savefig(fig_path + "FFT_1_" + str(d_obj) + ".png")
    if show_plot:
        plt.show()
    plt.close()

    plt.figure()
    plt.plot(freqs[pos_mask], dBs[pos_mask], c="SteelBlue", marker='.', label='Positive')  # plot positive part
    plt.plot(freqs[neg_mask], dBs[neg_mask], c="DarkRed", marker='.', label='Negative')  # plot negative part
    plt.xlabel('Frequency (Hz)')
    plt.ylabel("Amplitude in dBs")
    if d_obj == 1000:
        plt.xlim([1.85e9, 1.95e9])
        plt.ylim([-104, -97])
    else:
        plt.xlim([-0.5e9, 2e9])
    plt.title('Fast Fourier Transform of the Beat High Fr for ' + str(d_obj) + " m " +
        '{:.3f}'.format(freqs[max_index] * 10 ** (-6)) + " MHz")

    plt.legend()
    plt.grid(alpha=0.25, which='both')
    if save_plots:
        plt.savefig(fig_path + "FFT_2_" + str(d_obj) + ".png")
    if show_plot:
        plt.show()
    plt.close()

    # GENERAL PLOT OF FREQUENCIES N1
    plt.figure()
    plt.plot(t, A, c="DarkRed", alpha=0.6, marker='.', label=label_A)
    plt.plot(t, A2, c="Purple", alpha=0.6, marker='s', label=label_A2)
    plt.plot(t, echo, c="SeaGreen", alpha=0.6, marker='d', label=label_echo)
    plt.plot(t, y, c="SteelBlue", alpha=0.6, marker='.', label=FIR)
    if black_ground == 1:
        plt.plot(t, A, c="white", alpha=0.2, marker='.')
        plt.plot(t, A2, c="white", alpha=0.2, marker='s')
        plt.plot(t, echo, c="white", alpha=0.2, marker='d')
    plt.title("Linear Chirp, f(0)=" + str(f0 * 10 ** (-9)) + "GHz, f(1)=" + str(f1 * 10 ** (-9)) + "GHz")
    plt.xlabel('t (s), linear scale')
    plt.xscale('linear')
    plt.grid(alpha=0.25, which='both')
    plt.legend()
    if d_obj == 1000:
        plt.xlim([6.67e-6, 6.684e-6])
        plt.ylim([-1.05, 1.05])
    if save_plots:
        plt.savefig(fig_path + "chirp_Lineal-1-" + str(d_obj) + ".png")
    if show_plot:
        plt.show()
    plt.close()

    # GENERAL PLOT OF FREQUENCIES N2
    plt.figure()
    plt.plot(t, A, c="DarkRed", alpha=0.6, marker='.', label=label_A)
    plt.plot(t, A2, c="Purple", alpha=0.6, marker='s', label=label_A2)
    plt.plot(t, echo, c="SeaGreen", alpha=0.6, marker='d', label=label_echo)
    plt.plot(t, y, c="SteelBlue", alpha=0.6, marker='.', label=FIR)
    if black_ground == 1:
        plt.plot(t, A, c="white", alpha=0.2, marker='.')
        plt.plot(t, A2, c="white", alpha=0.2, marker='s')
        plt.plot(t, echo, c="white", alpha=0.2, marker='d')
    plt.title("Linear Chirp, f(0)=" + str(f0 * 10 ** (-9)) + "GHz, f(1)=" + str(f1 * 10 ** (-9)) + "GHz")
    plt.xlabel('t (s), linear scale')
    plt.xscale('linear')
    plt.grid(alpha=0.25)
    plt.legend()
    if d_obj == 1000:
        plt.xlim([6.6865e-6, 6.6985e-6])
        plt.ylim([-1.05, 1.05])
    if save_plots:
        plt.savefig(fig_path + "chirp_Lineal-2-" + str(d_obj) + ".png")
    if show_plot:
        plt.show()
    plt.close()

    # PLOT IN LOGARITHMIC X-AXIS SCALE
    plt.figure()
    plt.plot(t, A, c="DarkRed", alpha=1, marker='.', label=label_A)
    # plt.plot(t, A2, c="purple", alpha=0.8, marker='s', label=label_A2)
    # plt.plot(t, echo, c="green", alpha=0.7, marker='d', label=label_echo)
    if black_ground == 1:
        plt.plot(t, A, c="white", alpha=0.2, marker='.')
        # plt.plot(t, A2, c="white", alpha=0.2, marker='s')
        # plt.plot(t, echo, c="white", alpha=0.2, marker='d')
    plt.title("Linear Chirp, f(0)=" + str(f0 * 10 ** (-9)) + "GHz, f(1)=" + str(f1 * 10 ** (-9)) + "GHz")
    plt.xlabel('t (s), log scale')
    plt.xscale('log')
    plt.grid(alpha=0.25)
    plt.legend()
    # plt.xlim([6.67e-6, 6.684e-6])
    plt.ylim([-1.05, 1.05])
    plt.autoscale()
    if save_plots:
        plt.savefig(fig_path + "chirp_Log-" + str(d_obj) + ".png")
    if show_plot:
        plt.show()
    plt.close()

    # LINEAR RELATION
    plt.figure()
    plt.title("Linear relation, f(0)=" + str(f0 * 10 ** (-9)) + "GHz, f(1)=" + str(f1 * 10 ** (-9)) + "GHz")
    plt.plot(t, frequencies, c="SeaGreen", alpha=1, marker='^')
    if black_ground == 1:
        plt.plot(t, frequencies, c="white", alpha=0.25, marker='^')
    plt.grid(alpha=0.25)
    plt.ylabel('f (Hz)')
    plt.xlabel('t (s)')
    plt.autoscale()
    if save_plots:
        plt.savefig(fig_path + "chirp_Relation-" + str(d_obj) + ".png")
    if show_plot:
        plt.show()
    plt.close()

    plt.figure()
    plt.title("Linear relation, f(0)=" + str(f0 * 10 ** (-9)) + "GHz, f(1)=" + str(f1 * 10 ** (-9)) + "GHz")
    plt.plot(t, frequencies, c="SeaGreen", alpha=1, marker='^')
    if black_ground == 1:
        plt.plot(t, frequencies, c="white", alpha=0.25, marker='^')
    plt.grid(alpha=0.25)
    plt.ylabel('f (Hz)')
    plt.xlabel('t (s)')
    plt.autoscale()
    plt.xlim(np.min(t), 4e-8)
    plt.ylim(np.min(frequencies), 9.004e8)
    if save_plots:
        plt.savefig(fig_path + "chirp_stair" + str(d_obj) + ".png")
    if show_plot:
        plt.show()
    plt.close()

# DATAFRAME
end = time.time()
print("Building table of results...")
data = [['Peaks per step', peaks, ''],
        ['Initial frequency $F_{initial}$', f0 * 10 ** (-9), 'GHz'],
        ['Final frequency $F_{final}$', f1 * 10 ** (-9), 'GHz'],
        ['Minimum chirp duration', duration * 10 ** 9, 'ns'],
        ['Tracking distance $d_{max}$', distance, 'm'],
        ['Object distance for simulation $d_{obj}$', d_obj, 'm'],
        ['Resolution frequency $F_{res}$', f_res * 10 ** (-6), 'MHz'],
        ['Resolution time $T_{res}$', t_res * (10 ** 9), 'ns'],
        ['Resolution distance $d_{res}$', d_res, 'm'],
        ['Number of cloks', n_step, ''],
        ['$\\Delta_{F}$', delta_f * (10 ** (-6)), 'MHz'],
        ['Number of discrete points $n$', n, ''],
        ['Time of flight to the object and back $TOF$', TOF * (10 ** 9), 'ns'],
        ['Cut-off frequency $F_{c}$', fc * 10 ** (-6), 'MHz'],
        ['Distance of the object to avoid time shift of the wave', dist2 / 2, 'm'],
        ['Time resolution ratio $\\frac{T_{dist}}{T_{res}}$', TOF / t_res, ''],
        ['Time to phase shift signal by 1/2 of $T_{res}$', d_phase_shift, 's'],
        ["FFT Frequency: ", freqs[max_index] * 10 ** (-3), 'kHz'],
        ["Computational time: ", end - start, 's']
        ]

df = pd.DataFrame(data, columns=['Description', 'Variable', 'Units'])

df.to_excel('chirp_simulation.xlsx')
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
print("Table of results:")
print("")
print("\\begin{table}[h]")
print("\\centering")
print("\\begin{center}")
print(latex_table)
print("\\end{center}")
print("\\caption{Chirp Emission and Comparison Simulation Values}")
print("\\end{table}")
print("")

with open(fig_path + "Chirp_result-" + str(d_obj) + ".txt", 'w') as f:
    f.write('\n')
    f.write('Table of results:\n')
    f.write('\n')
    f.write('\\begin{table}[h]\n')
    f.write('\\centering\n')
    f.write('\\begin{center}\n')
    f.write(latex_table + '\n')
    f.write('\\end{center}\n')
    f.write('\\caption{Chirp Emission and Comparison Simulation Values}\n')
    f.write('\\end{table}\n')
    f.write('\n')

print("Table of results DONE!")
print("Program finished")
