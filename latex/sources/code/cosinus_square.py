import numpy as np
import matplotlib.pyplot as plt
import math

# This script is used to understand the square of a cosine
res = 1000
t0 = 50
t1 = 90
t = np.linspace(t0, t1, res)
A = np.zeros_like(t)
E = np.zeros_like(t)


def square_cosine(x):
    return 0.5 * (math.cos(2 * x) + 1)


for i in range(len(A)):
    A[i] = square_cosine(t[i])
    E[i] = math.cos(t[i])


fft_A2 = np.fft.fft(A)
fft_f = np.fft.fft(t)
freqs = np.fft.fftfreq(len(A), d=res)

# FFT
plt.plot(freqs, np.abs(fft_A2))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
#plt.xlim([-2e-9, 2e-9])
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.show()

# PLOT SECTION
plt.style.use("dark_background")
plt.plot(t, A, c="green", alpha=1, marker='.')
plt.plot(t, E, c="blue", alpha=1, marker='.')
plt.title("cos² ")
plt.xlabel('x')
plt.ylabel('y')
plt.xscale('linear')
plt.grid(alpha=0.25)
plt.show()
