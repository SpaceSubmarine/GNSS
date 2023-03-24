import matplotlib.pyplot as plt
from scipy.signal import chirp
import numpy as np

t = np.linspace(0, 10, 1500)
w = chirp(t, f0=0.9, f1=5, t1=10, method='linear', phi=270)
plt.plot(t, w, c="DarkCyan", alpha=0.6, marker='.')
plt.title("Linear Chirp, f(0)=0.9, f(10)=5")
plt.xlabel('t (sec)')
plt.show()
