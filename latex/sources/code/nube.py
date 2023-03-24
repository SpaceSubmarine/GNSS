import numpy as np
import matplotlib.pyplot as plt


N = 10000
opcion = 1

if opcion == 1:
    plt.style.use("dark_background")
    # Genera algunos datos aleatorios
    x = np.random.randn(N)
    y = np.random.randn(N)
    z = np.random.randn(N)

    # Crea una figura y un eje 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Crea una nube de densidades
    ax.scatter(x, y, z, c=z, cmap='inferno')

    # Muestra la figura
    plt.show()
    print("done")




if opcion == 2:
    # Crear una matriz 3D cubica con valores aleatorios
    matriz_3D = np.random.rand(N, N, N)

    # Obtener los valores x, y, z y c (color) a partir de la matriz 3D
    x, y, z = np.indices((N, N, N))
    c = matriz_3D

    plt.style.use("dark_background")
    # cmap = plt.colorbar()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=c, cmap='rainbow')
    plt.show()



'''import matplotlib.pyplot as plt
import numpy as np

# Genera algunos datos aleatorios
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)

# Crea una figura y un eje 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crea una nube de densidades
ax.scatter(x, y, z, c=z, cmap='rainbow')

# Muestra la figura
plt.show()
'''