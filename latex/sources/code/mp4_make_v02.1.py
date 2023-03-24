import cv2
import os
import time
inicio = time.time()
fps = 25  # fotogramas por segundo 60 para video y 120 para simulacion con 240fps ori
nombre = '221230-Lab_s_V04.mp4'
#nombre = 'MatrizCV.mp4'
# folder path
#dir_path = './frames/V_Generado/'
dir_path = './Frame_generation/opencv/Frames/'
#dir_path = './Frame_generation/opencv/TVL1/'
#dir_path = './Frame_generation/MatPlot/V_field/'
#dir_path = './videos/videos_laboratorio/frames/'
#dir_path = './Frame_generation/opencv/LK2/'
count = 0

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
print('File count:', count)

################################################################
# path = './frames/opencv/*.png'
img_array = []

for i in range(1, count):
    filename = str(i)+'.png'
    img = cv2.imread(str(dir_path + filename))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(nombre, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

fin = time.time()
print("Tiempo de ejecución: ", fin-inicio)
