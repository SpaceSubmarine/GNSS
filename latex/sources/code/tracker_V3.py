from pathlib import Path
from time import sleep
import pandas as pd
import numpy as np
import time
import glob
import cv2
import os
inicio = time.time()
import sys



# INPUTS
path = 0#"./videos/particulas_slow.mp4"
tol = 2  # Tolerancia 0 con videos grandes (distancia entre las aristas del recuadro y el contorno)
area_input = 25  # 200 es recomendado para videos y 0 para la particula
escalado = 1  # Escala para la visualización del video
contornos = True  # Visualizacion de los contornos
recuadros = True  # Visualizacion de recuadros
velocidad = False
limpiar = False
noise_red = 50  # Reduccion de ruido del ThresHold (50:particula. Mayor numero, menor ruido)
connection_type = 4  # tipo de conectividad entre componentes
filepath = Path('./DataFrame.csv')  # para guardar el arxivo CSV
track_election = -2  # ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']



# Limpieza de datos==============================================================================
files = glob.glob('./datos/DF_Values/*')
files1 = glob.glob('./datos/DF_Labels/*')
files2 = glob.glob('./frames/opencv/*')
if limpiar:
    for f in files:
        os.remove(f)  # Limpia la carpeta DF_Values
    for f in files1:
        os.remove(f)  # Limpia la carpeta DF_Labels
    for f in files2:
        os.remove(f)  # Limpia la carpeta opencv (frames de tracking)

# Definimos la funcion para reescalar la imagen para poder ver mejor el pixel


def rescaleframe(frame, scale=escalado):  # Tener en cuenta "shadowing" y "global variables"
    width = int(frame.shape[1] * scale)
    heigh = int(frame.shape[0] * scale)
    dimensions = (width, heigh)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Object detection (desde una camara fija, en este caso solo se mueve el pixel)
# el arg para evitar falsos positivos
thresh_function = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=noise_red)

# ===============================================================================================
df_centroids = []  # Inicializcion del data frame de los centroides
i = 0  # para saber el fotograma
pxold = [0, 0, 0, 0, 0, 0, 0]
N_fot = 0
values_old = pd.DataFrame(np.zeros(50))


# TRACKER

# esto es para ver las versiones del open CV
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
tracker_type = tracker_types[-3]

if tracker_type == 'BOOSTING':
    print("boost")
    tracker = cv2.legacy.TrackerBoosting_create()
if tracker_type == 'MIL':
    print("mil")
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':  # se observa que al seguir una particula que cambia de tamaño, la pierde
    print("KCF")
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':  # funciona muy mal, no interesa
    print("TDL")
    tracker = cv2.legacy.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':  # es rapido y sigue bastante bien el objeto, cuando lo pierde, lo pierde
    print("MEDIANFLOW")
    tracker = cv2.legacy.TrackerMedianFlow_create()
if tracker_type == 'CSRT':  # es un poco mas lento, funciona muy bien, cuando pierde el objeto, intenta buscar otro igual
    print("CSRT")
    tracker = cv2.TrackerCSRT_create()
if tracker_type == 'MOSSE':
    print("MOSSE")
    tracker = cv2.legacy.TrackerMOSSE_create()

# Read video
cap = cv2.VideoCapture(path)

# Exit if video not opened.
if not cap.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = cap.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

# Define an initial bounding box
#bbox = (287, 23, 86, 320)

# Uncomment the line below to select a different bounding box
frame = cv2.flip(frame, 1)
bbox = cv2.selectROI(frame, False)
print(bbox)
# Initialize tracker with first frame and bounding box
# mirar de introducir en el recuadro
# visualizacon del video=========================================================================
while True:
    ret, frame_original = cap.read()
    frame = frame_original
    frame = rescaleframe(frame)
    height, width, _ = frame.shape

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok = tracker.init(frame)
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break