import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import cv2
import os
import time
inicio = time.time()
# Inputs=========================================================================================
# se introducen los valores para ciertos parametros variables de las imagenes que se muestran
# el campo de velocidades
frame_shape_path = cv2.imread("./Frame_generation/opencv/Frames/1.png")
dimensions = frame_shape_path.shape  # Image Height, Image Width, Number of Channels
redux = 5  # Factor de reduccion 16
t_range = 1500000
t_redux = 0.96
escala_flecha = (redux*4)/50  # se cambia el escalado de la flecha en funcion de redux
guardar_png = False
limpiar_png = False
solido = True
vmin = 0
vmax = 8  # este valor debe ser igual al de la distancia de deteccion
# Contar el numero de archivos CSV ==============================================================
isfile = os.path.isfile
join = os.path.join
directory = './datos/DF_Values/'
number_of_files = sum(1 for item in os.listdir(directory) if isfile(join(directory, item)))
DF_Vx_path = "./datos/dfvx/"  # path de los DataFrames de la velocidad en X

# ===============================================================================================
if limpiar_png:
    files = glob.glob('./Frame_generation/MatPlot/V_field/*')
    for f in files:
        os.remove(f)  # Limpia la carpeta V_field

# Step 1=========================================================================================
# Crear las matrices donde se alojan las velocidades en las posiciones dadas por los centroides
# Inicialización de variables:
Mx, My = dimensions[0]/redux, dimensions[1]/redux
Vt_old = np.zeros((int(Mx), int(My)))
tv = np.zeros((int(Mx), int(My)))

for n_files in range(number_of_files-1):
    print(n_files)
    csv_path = "./datos/DF_Values/DF_Values_F" + str(n_files) + ".csv"
    df = pd.read_csv(csv_path)  # se lee el archivo de la iteracion con pandas
    # Clean DataFrame============================================================================
    del df['ID']  # no se necesitan los identificadores en este programa
    del df["Unnamed: 0"]  # la primera columna es un indice que no tiene utilidad ni nombre4
    # Inicializacion
    Px = []
    Py = []
    Vx = []
    Vy = []
    Vt = []

    Vx_matrix = np.zeros([dimensions[0], dimensions[1]])  # Matriz Vx del tamaño de la imagen
    Vy_matrix = np.zeros([dimensions[0], dimensions[1]])
    Vt_matrix = np.zeros([dimensions[0], dimensions[1]])

    # Generacion de matrices
    for i_df in range(len(df)):
        Px.append(df.loc[i_df, "Px"])
        Py.append(df.loc[i_df, "Py"])
        Vx.append(df.loc[i_df, "Vx"])
        Vy.append(df.loc[i_df, "Vy"])
        Vt.append(df.loc[i_df, "VT"])
        Vx_matrix[Py[i_df], Px[i_df]] = -Vx[i_df]
        Vy_matrix[Py[i_df], Px[i_df]] = -Vy[i_df]
        Vt_matrix[Py[i_df], Px[i_df]] = Vt[i_df]

    # Utilizando el factor de reducción, se generan las matrices
    #Mx, My = dimensions[0]/redux, dimensions[1]/redux
    Mx_redux = np.zeros((int(Mx), int(My)))
    My_redux = np.zeros((int(Mx), int(My)))
    Vt_redux = np.zeros((int(Mx), int(My)))


    Rx, Ry = Mx_redux.shape

    for i in range(Rx):
        for j in range(Ry):
            var1 = int(i*redux)
            var11 = int((i+1)*redux)
            var2 = int(j * redux)
            var22 = int((j+1) * redux)
            MMx = np.sum(Vx_matrix[var1:var11, var2:var22])
            MMy = np.sum(Vy_matrix[var1:var11, var2:var22])
            Vtt = np.sum(Vt_matrix[var1:var11, var2:var22])
            elements_redux = np.size(MMx)
            Mx_redux[i, j] = MMx/elements_redux
            My_redux[i, j] = MMy/elements_redux
            Vt_redux[i, j] = Vtt/elements_redux

            # Limpiar outliers
            if Vt_redux[i, j] > vmax*0.95:
                Vt_redux[i, j] = 0
                Mx_redux[i, j] = 0
                My_redux[i, j] = 0

            if Vt_redux[i, j] == 0:
                if tv[i, j] > t_range:
                    tv[i, j] = 0
                    Vt_redux[i, j] = 0
                else:
                    Vt_redux[i, j] = Vt_old[i, j]
                    if solido == True:
                        Vt_redux[i, j] = Vt_old[i, j]*t_redux
                    tv[i, j] = tv[i, j] + 1
            else:
                tv[i, j] = 0
            Vt_old[i, j] = Vt_redux[i, j].copy()

    # Campo de velocidadees temporal ============================================================
    plt.clf()
    plt.ion()
    # scale_units="inches", scale=50

    # para escoger colores, utilizar el siguiente enlace:
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # vmin=0, vmax=20 interpolation="gaussian", "lanczos", cmap="jet","inferno", "YlOrRd"
    plt.imshow(Vt_redux, cmap="inferno", interpolation="gaussian", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.set_label('V(pixels/frame)', rotation=270)
    #cbar.set_ticklabels(update_ticks=True)
    plt.quiver(Mx_redux, My_redux, scale=escala_flecha, color="w", scale_units="xy",  minlength=0.001, angles='xy')
    # plt.quiver(Vx_matrix, Vy_matrix, scale=escala_flecha,  scale_units="xy",  minlength=0.0001, angles='xy')
    plt.title("Frame:" + str(n_files))
    plt.xlabel("Px")
    plt.ylabel("Py")
    plt.xlim(0, dimensions[1]/redux)
    plt.ylim(dimensions[0]/redux, 0)
    plt.show()
    plt.pause(0.00000000000000001)
    if guardar_png:
        plt.savefig("./Frame_generation/MatPlot/V_field/" + str(n_files) + ".png")
    plt.clf()

fin = time.time()
print("Tiempo de ejecución: ", fin-inicio)
