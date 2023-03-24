import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import numpy as np
import time
import re
import csv
start = time.time()

file_csv = '4k_0.01us_20MHz.csv'
file_txt = '4k_0.1us_10MHz.txt'
path = './med_osci/'
print("for txt file press '0' and for csv files press '1'")
type_file = int(input())


if type_file == int(1):
    file = file_csv
    data_file = "csv"
    match_1 = re.search(r'(\d+(\.\d+)?)us', file)  # floats before us
    match_2 = re.search(r'(\d+(\.\d+)?)MHz.', file)  # floats before MHz
    T_div = float(match_1.group(1))*10**(-3)  # us time division of oscilloscope
else:
    file = file_txt
    data_file = "txt"
    match_1 = re.search(r'(\d+(\.\d+)?)us', file)  # floats before ns
    match_2 = re.search(r'(\d+(\.\d+)?)MHz.', file)  # floats before MHz
    T_div = float(match_1.group(1))  # us time division of oscilloscope

k_res = re.search(r'(\d+(\.\d+)?)k', file)  # floats before us
k_res = int(k_res.group(1))
n_points = 1024 * k_res


rel = 0.0625  # relation between time divisions and time recorded
F_inp = float(match_2.group(1))  # MHz
'''T_record = T_div/rel
T_res = T_record/n_points'''

T_res = T_div * 4 * k_res
T_record = T_res

if data_file == "txt":
    # Leer el archivo de texto línea por línea y guardar los valores en una lista
    with open(path + file, 'r') as f:
        lines = f.readlines()[4:]
        values_1 = [float(line.strip()) for line in lines if line.strip()]


if data_file == "csv":
    with open(path+file, mode='r') as file:
        reader = csv.DictReader(file)

        # Crear una lista vacía para almacenar los datos de la columna 'CH1'
        time_line = []
        values_1 = []
        values_2 = []
        values_3 = []
        values_4 = []
        # Iterar sobre las filas del archivo CSV y agregar los datos de la columna 'CH1' a la lista
        for row in reader:
            time_line.append(float(row['time']))
            values_1.append(float(row['CH1']))
            values_2.append(float(row['CH2']))
            values_3.append(float(row['CH3']))
            values_4.append(float(row['CH4']))

# Crear un DataFrame de pandas a partir de la lista de valores
'''df = pd.DataFrame(values_1, columns=['Valor'])
print(df)
# Convertir el DataFrame de pandas a un array de numpy
V = df['Valor'].values
t = np.linspace(0, T_record, n_points) * 10**(-6)'''
V = values_1

if data_file == "csv":
    t = time_line
else:
    t = np.linspace(0, T_record, n_points) * 10**(-6)

# Define the filter window
window = np.ones(5)/5  # This creates a filter window of 5 values, all set to 1/5
# Apply the filter to the signal using convolution
V_filtered = np.convolve(V, window, mode='same')
# This applies the filter window to the signal V using convolution
# and stores the filtered signal in V_filtered
# The 'same' mode ensures that the filtered signal has the same length as the original
# Compute the FFT of the filtered signal
fft_V = np.fft.fft(V_filtered)
# This computes the FFT of the filtered signal and stores the result in fft_V
# The filtered signal is used as the input signal to the FFT function
freqs = np.fft.fftfreq(len(V_filtered), d=t[1] - t[0])
amplitud = 2.0 / (len(t)) * abs(fft_V)  # compute the amplitude
pos_mask = freqs > 0  # positive filter
neg_mask = freqs < 0  # negative filter

# this commented block is before the applied V_filtered correction
'''# Fast Fourier Transform
print("Doing FFT...")
fft_V = np.fft.fft(V)
freqs = np.fft.fftfreq(len(V), d=t[1] - t[0])
amplitud = 2.0 / (len(t)) * abs(fft_V)  # compute the amplitude
pos_mask = freqs > 0  # positive filter
neg_mask = freqs < 0  # negative filter
'''

max_value = np.max(amplitud)  # encuentra el valor máximo en el array
max_index = np.argmax(amplitud)  # encuentra la posición del valor máximo en el array
FFT_F = abs(freqs[max_index]*10**(-6))  # FFT in MHz
F_error = (F_inp - FFT_F)*10**6  # FFT error in Hz


# DATAFRAME
end = time.time()
print("Building table of results...")
data = [['Generated Frequency $F_{GEN}$', F_inp, '$MHz$'],
        ['FFT Freq $F_{FFT}$', FFT_F, '$MHz$'],
        ['FFT Freq error $E_{FFT}$', F_error, '$Hz$'],
        ['Number of data-points $n$', n_points, ''],
        ['Time per division (oscilloscope) $T_{div}$', T_div, '$\mu s$'],
        ['Time resolution $T_{res}$', T_res, '$\mu s$'],
        ['Recorded time $T_{rec}$', T_record, '$\mu s$'],
        ["Computational time: ", end - start, 's']
        ]

df = pd.DataFrame(data, columns=['Description', 'Variable', 'Units'])

df.to_excel('chirp_simulation.xlsx')
latex_table = tabulate(df, headers='keys', tablefmt='latex', floatfmt='.4E')
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
print("\\caption{FFT with data from oscilloscope for ", str(F_inp), "MHz with a $T_{div}$ of ", str(T_div), "$\mu s$}")
print("\\end{table}")
print("")


# GENERAL PLOT OF FREQUENCIES N1
plt.figure()
plt.plot(t, V, c="SteelBlue", alpha=0.6, marker='.')
plt.xlabel('t (s), linear scale')
plt.xscale('linear')
plt.grid(alpha=0.25, which='both')
plt.legend()
plt.show()
plt.close()


dBs = []
for i in range(len(amplitud)):
    dBs.append(20 * np.log(np.abs(amplitud[i])))
dBs = np.array(dBs)

plt.figure()
plt.plot(freqs[pos_mask], dBs[pos_mask], c="SteelBlue", label='Positive', marker='d')  # plot positive part
plt.plot(freqs[neg_mask], dBs[neg_mask], c="DarkRed", label='Negative', marker='d')  # plot negative part
plt.grid(alpha=0.25, which='both')
plt.xlabel('Frequency (Hz)')
plt.ylabel("Amplitude in dBs")
plt.legend()
plt.show()


