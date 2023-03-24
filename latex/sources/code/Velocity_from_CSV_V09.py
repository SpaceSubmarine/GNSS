import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import cv2
import os
import time
start = time.time()
# Inputs
# values are introduced for certain variable parameters of the images that are displayed
# the field of velocities
frame_shape_path = cv2.imread("./Frame_generation/opencv/Frames/1.png")
dimensions = frame_shape_path.shape  # # Image Height, Image, Number of Channels
redux = 5   # Reduction factor 16
t_range = 1500000
t_redux = 0.96
arrow_scale = (redux*4)/50  # the arrow scaling is changed according to redux
save_png = False
clean_png = False
solid = True
vmin = 0
vmax = 8  # this value should be equal to the detection distance
# Count the number of CSV files
isfile = os.path.isfile
join = os.path.join
directory = './datos/DF_Values/'
number_of_files = sum(1 for item in os.listdir(directory) if isfile(join(directory, item)))
DF_Vx_path = "./datos/dfvx/"  # path of the velocity DataFrames in X


if clean_png:
    files = glob.glob('./Frame_generation/MatPlot/V_field/*')
    for f in files:
        os.remove(f)  # Clean up the V_field folder

# Step 1
# Create the matrices where the velocities at the positions given by the centroids are housed
# Variable initialization:
Mx, My = dimensions[0]/redux, dimensions[1]/redux
Vt_old = np.zeros((int(Mx), int(My)))
tv = np.zeros((int(Mx), int(My)))

for n_files in range(number_of_files-1):
    print(n_files)
    csv_path = "./datos/DF_Values/DF_Values_F" + str(n_files) + ".csv"
    df = pd.read_csv(csv_path)  # the file of the iteration is read with pandas
    # Clean DataFrame
    del df['ID']  # the identifiers are not needed in this program
    del df["Unnamed: 0"]  # the first column is an index that has no utility or name
    # Initialization
    Px = []
    Py = []
    Vx = []
    Vy = []
    Vt = []

    Vx_matrix = np.zeros([dimensions[0], dimensions[1]])  # Matrix Vx of the size of the image
    Vy_matrix = np.zeros([dimensions[0], dimensions[1]])  # Matrix Vx of the size of the image
    Vt_matrix = np.zeros([dimensions[0], dimensions[1]])  # Matrix Vt of the time of the image

    # Generation of matrices
    for i_df in range(len(df)):
        Px.append(df.loc[i_df, "Px"])
        Py.append(df.loc[i_df, "Py"])
        Vx.append(df.loc[i_df, "Vx"])
        Vy.append(df.loc[i_df, "Vy"])
        Vt.append(df.loc[i_df, "VT"])
        Vx_matrix[Py[i_df], Px[i_df]] = -Vx[i_df]
        Vy_matrix[Py[i_df], Px[i_df]] = -Vy[i_df]
        Vt_matrix[Py[i_df], Px[i_df]] = Vt[i_df]

    # Using the reduction factor, the matrices are generated
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

            # Clean outliers
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
                    if solid == True:
                        Vt_redux[i, j] = Vt_old[i, j]*t_redux
                    tv[i, j] = tv[i, j] + 1
            else:
                tv[i, j] = 0
            Vt_old[i, j] = Vt_redux[i, j].copy()

    # Temporal velocity field
    plt.clf()
    plt.ion()
    # scale_units="inches", scale=50

    # to choose colors, use the following link:
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # vmin=0, vmax=20 interpolation="gaussian", "lanczos", cmap="jet","inferno", "YlOrRd"
    plt.imshow(Vt_redux, cmap="inferno", interpolation="gaussian", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.set_label('V(pixels/frame)', rotation=270)
    #cbar.set_ticklabels(update_ticks=True)
    plt.quiver(Mx_redux, My_redux, scale=arrow_scale, color="w", scale_units="xy",  minlength=0.001, angles='xy')
    # plt.quiver(Vx_matrix, Vy_matrix, scale=arrow_scale,  scale_units="xy",  minlength=0.0001, angles='xy')
    plt.title("Frame:" + str(n_files))
    plt.xlabel("Px")
    plt.ylabel("Py")
    plt.xlim(0, dimensions[1]/redux)
    plt.ylim(dimensions[0]/redux, 0)
    plt.show()
    plt.pause(0.00000000000000001)
    if save_png:
        plt.savefig("./Frame_generation/MatPlot/V_field/" + str(n_files) + ".png")
    plt.clf()

end = time.time()
print("Computational time: ", end-start)
