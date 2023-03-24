from pathlib import Path
import pandas as pd
import numpy as np
import math
import time
import glob
import cv2
import os

inicio = time.time()
# DESCRIPCION: Este script realiza el tracking de objetos.=======================================
# Es capaz de almacenar los datos en CSV para cada frame de las posiciones de los centroides
# por cada pixel y generar DataFrames con los identificadores de los objetos con su posicion, velocidad
# y fotograma. Estos datos deben ser exportados para cada fotograma, de esta manera, un script con
# matplotlib.pyplot pueda generar matrices de velocidades en X e Y con el fin de graficar esas velocidades
# para cada instante de tiempo y así generar una animación transitoria con opencv o con matplotlib.

# Inputs=========================================================================================
tol = 2  # Tolerancia 0 con videos grandes (distancia entre las aristas del recuadro y el contorno)
print("HSV = Pulse 1 ")
print("GRAY = Pulse 0 ")
HSV = int(input())  # 200 es recomendado para videos y 0 para la particula
print("Introduzca el área mínima en pixels:")
area_low = int(input())  # 200 es recomendado para videos y 0 para la particula
print("Introduzca el área máxima en pixels:")
area_high = int(input())  # 100000
print("Introduzca el factor de escala:")
escalado = float(input())  # Escala para la visualización del video
print("Introduzca la distancia de seguimiento en pixels:")
dist = int(input())  # escalado * 6  # distancia en pixels
contornos = False  # Visualizacion de los contornos
recuadros = True  # Visualizacion de recuadros
paso_a_paso = False

print("¿Desea guardar bases de datos?:")
print("SI = Pulse 1 ")
print("NO = Pulse 0 ")
guardar_datos = int(input())
ver_velocidades = False
trackerCV = False  # Activa o desactiva el tracker que previamente seleccionado
Show_ROI = False
Show_Frames = True
Show_ThreshHold = False
Show_MASK_conn = False
print("¿Desea guardar los fotogramas generados?:")
print("SI = Pulse 1 ")
print("NO = Pulse 0 ")
Save_Frames = int(input())
Save_thresh = False
Save_conn = False
limpiar_frames = True
limpiar_datos = True
recorte = False
noise_red = 50  # Reduccion de r1uido del ThresHold (50:particula. Mayor numero, menor ruido)
connection_type = 4  # tipo de conectividad entre componentes
hmin = 165  # 20  # 0
smin = 0  # 137  #0
vmin = 22  # 58  #255
hmax = 185  # 41  #90
smax = 252  # 255  #28
vmax = 255  # 255  #255
e = 0

# Colores========================================================================================
# para poder ajustar los colores sin tener que hacerlo para cada putText
blue = (255, 0, 0)
red = (0, 0, 255)
orange = (0, 150, 255)

# ==============================================================================================
# path = "./videos/videos_laboratorio/particulas_slow.mp4"
path = './videos/videos_laboratorio/SnapSave.io-The Battleship Texas_ Aerial view of famed ship cruising to Galveston.mp4'
# Limpieza de datos==============================================================================
if limpiar_datos:
    files = glob.glob('./datos/DF_Values/*')
    files1 = glob.glob('./datos/DF_Labels/*')
    for f in files:
        os.remove(f)  # Limpia la carpeta DF_Values
    for f in files1:
        os.remove(f)  # Limpia la carpeta DF_Labels
if limpiar_frames:
    files2 = glob.glob('./Frame_generation/opencv/Frames/*')
    files3 = glob.glob('./Frame_generation/opencv/Mask_thresh/*')
    files4 = glob.glob('./Frame_generation/opencv/Mask_Conn/*')
    for f in files2:
        os.remove(f)  # Limpia la carpeta opencv (Frames)
    for f in files3:
        os.remove(f)  # Limpia la carpeta opencv (Mask_thresh)
    for f in files4:
        os.remove(f)  # Limpia la carpeta opencv (Mask_Conn)


# ===============================================================================================
# Definimos la funcion para reescalar la imagen para poder ver mejor el pixel
def rescaleframe(frame, scale=escalado):  # Tener en cuenta "shadowing" y "global variable"
    width = int(frame.shape[1] * scale)
    heigh = int(frame.shape[0] * scale)
    dimensions = (width, heigh)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


# Inicializando frames
cap = cv2.VideoCapture(path)  # 0 si se quiere capturar webcam
thresh_function = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=noise_red, detectShadows=False)

# Inicializaciones===============================================================================
df_centroids = []  # Inicializcion del data frame de los centroides
N_frame = 0  # para saber el fotograma
values_old = pd.DataFrame(np.zeros(50))  # inicializacion del df valores anteriores
center_points_prev_frame = []  # inicializacion centros fotograma anterior
tracking_objects = {}  # inicializacion objetos tracking
track_id = 0  # inicializacion numero de ID del objeto
center_points = []  # inicializamos lista de puntos

# Visualizacion del video=========================================================================
while True:
    ret, frame_original = cap.read()
    # si hacemos un test con la camara, sirve para verlo con mirror
    if path == 0:
        frame_original = cv2.flip(frame_original, 1)
    if not ret:
        break

    # Si se selecciona el recorte
    if recorte:
        frame_original = frame_original[33:1400, 200:1809]
    frame_original = rescaleframe(frame_original)
    frame = frame_original.copy()

    # HSV==========================================================================
    # arrays con valores máximos y mínimos de HSV
    lower_hsv = np.array([hmin, smin, vmin])
    upper_hsv = np.array([hmax, smax, vmax])

    # Cambio de formato de BGR a HSV (recuerda de volver a convertirlo a BGR después)
    if HSV:
        hsv = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_BGR2HSV)
        MASK_Color = cv2.inRange(hsv, lower_hsv, upper_hsv)
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        MASK_Color = hsv

    # =============================================================================
    height, width, _ = frame.shape  # height, width, _ = frame.shape     (con colores)

    # point current frame
    center_points_cur_frame = []

    # Object detection
    thresh = thresh_function.apply(MASK_Color)
    # cv2.imwrite('./testing/MASK_thresh/' + str(N_frame) + '.png', thresh)

    # ===============================================================================================
    # Connected Components Analysis function
    totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(thresh, connection_type, cv2.CV_32S)
    MASK_Conn = np.zeros(label_ids.shape, dtype="uint8")
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]
        if (area > area_low) and (area < area_high):
            componentMask = (label_ids == i).astype("uint8") * 255
            MASK_Conn = cv2.bitwise_or(MASK_Conn, componentMask)
    totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(MASK_Conn, connection_type, cv2.CV_32S)
    centroid = np.ndarray.round(centroid).astype(int)
    df_centroids.append(centroid)

    # ================================================================================================
    # Generacion de dataframes por fotograma
    values = pd.DataFrame(values)  # Este data frame guarda valores numericos para cada label
    filepath_values = Path('./datos/DF_Values/DF_Values_F' + str(N_frame) + '.csv')
    filepath_values.parent.mkdir(parents=True, exist_ok=True)
    values["Vx"] = np.nan  # inicializacion de la columna del df por fotograma
    values["Vy"] = np.nan  # inicializacion de la columna del df por fotograma
    values["VT"] = np.nan  # inicializacion de la columna del df por fotograma
    label_ids = pd.DataFrame(label_ids)  # Este data frame es una matriz donde los valres son los labels
    filepath_labels = Path('.//datos/DF_Labels/DF_Labels_F' + str(N_frame) + '.csv')
    filepath_labels.parent.mkdir(parents=True, exist_ok=True)

    # ===============================================================================================
    # region de interes dentro del video
    roi = frame[0:height, 0:width]  # Es la zona en la que se desea graficar los contornos
    # Se utiliza la mascara binaria (thresh) para hacer el 'Contours'
    detections = []
    MASK_BOX = frame
    ROI_number = 0
    N_boxes = 0
    cnts = cv2.findContours(MASK_Conn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ROI = frame[y:y + h, x:x + w]

        # cv2.imwrite('./testing/MASK_BOX/' + str(N_frame) + '.png', MASK_BOX)
        ROI_number += 1
        if contornos:  # Si se activan los contornos, se grafican:
            # Definicion de los contornos para cada cluster aceptado
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)  # (tipo,(B, G, R), espesor)

        cx = int((x + x + w) / 2)  # coordenada x del centro de la caja
        cy = int((y + y + h) / 2)  # coordenada y del centro de la caja
        center_points_cur_frame.append((cx, cy))  # concatenamos el nuevo punto
        # X,Y coordenadas esquina + width(ancho) y height(altura)
        detections.append([x, y, w, h])
        # Definicion de etiquetas de los recuadros en el video

        recuadro_actual = str(label_ids.loc[int(str(c[0][0][1])), int(str(c[0][0][0]))])
        if recuadros:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 210), 1)
            cv2.putText(roi, "X:" + str(cx) + " Y:" + str(cy), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 150, 255))

    # TRACKING=======================================================================================
    max_vt = [0]

    dfValues = []
    if N_frame <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < dist:
                    tracking_objects[track_id] = pt
                    track_id += 1

    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exist = False
            for pt in center_points_cur_frame_copy:
                distance = ((pt2[0] - pt[0]) ** 2 + (pt2[1] - pt[1]) ** 2) ** (1 / 2)
                VT = distance  # pixeles por fotograma
                vx = pt2[0] - pt[0]  # pixeles por fotograma
                vy = pt2[1] - pt[1]  # pixeles por fotograma

                max_vt.append(int(VT))
                if distance < dist:
                    if pt in center_points_cur_frame:
                        tracking_objects[object_id] = pt
                        object_exist = True
                        center_points_cur_frame.remove(pt)
                        cv2.putText(frame, str(object_id), pt, cv2.FONT_HERSHEY_DUPLEX, 0.4, blue, 1)
                        cv2.circle(frame, pt, 1, (0, 170, 0), -1)
                        dfValues.append([object_id, pt2[0], pt2[1], vx, vy, VT])
                        if ver_velocidades:
                            VT = "V: " + str("%.2f" % round(VT, 2))
                            Vx = "Vx: " + str(pt2[0] - pt[0])
                            Vy = "Vy: " + str(pt2[1] - pt[1])
                            cv2.putText(frame, VT, (pt2[0], pt2[1] + 10), cv2.FONT_HERSHEY_DUPLEX, 0.35, red, 1)
                            cv2.putText(frame, (Vx + " " + Vy), (pt2[0], pt2[1] + 25), cv2.FONT_HERSHEY_DUPLEX, 0.35,
                                        red, 1)
                    continue

            # Remover IDs perdidos
            if not object_exist:
                tracking_objects.pop(object_id)

        # añade nuevos ID
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    # ===============================================================================================
    N_frame += 1
    dfValues = pd.DataFrame(dfValues, columns=['ID', 'Px', 'Py', 'Vx', 'Vy', 'VT'])
    # Guardado de bases de datos
    if guardar_datos:
        label_ids.to_csv(filepath_labels)
        dfValues.to_csv(filepath_values)

    # values_old = values
    min_vt = np.min(np.array(max_vt))
    max_vt = np.max(np.array(max_vt))
    median_vt = (max_vt - min_vt) / 2

    # Textos informativos por cada frame=============================================================
    cv2.putText(frame, "AD Tracker", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
    cv2.putText(frame, "Clusters : " + str(int(totalLabels)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, orange, 2)
    cv2.putText(frame, "Frame : " + str(N_frame), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, orange, 2)
    cv2.putText(frame, "Escalado : " + str(escalado), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, orange, 2)

    # Visualizaciones durante la ejecucion===========================================================
    if Show_ROI:
        cv2.imshow("Roi", roi)  # Para visualizar el video de solo la region de interes
    if Show_Frames:
        cv2.imshow("TRACKING", frame)  # Para visualizar el video con contornos y recuadros
    if Show_ThreshHold:
        cv2.imshow("ThresHolding IMAGE", thresh)  # Para visualizar el video del ThresHolding
    if Show_MASK_conn:
        cv2.imshow("MASK_Conn IMAGE", MASK_Conn)  # Para visualizar el video del ThresHolding
    if Save_Frames:
        cv2.imwrite('./Frame_generation/opencv/Frames/' + str(N_frame) + '.png', frame)
    if Save_thresh:
        cv2.imwrite('./Frame_generation/opencv/Mask_thresh/' + str(N_frame) + '.png', thresh)
    if Save_conn:
        cv2.imwrite('./Frame_generation/opencv/Mask_Conn/' + str(N_frame) + '.png', MASK_Conn)
    print(N_frame)

    # copia de los puntos
    center_points_prev_frame = center_points_cur_frame.copy()

    # Eleccion de visualizar el video normal o pasando fotogramas manualmente
    if paso_a_paso:
        key = cv2.waitKey(0)  # Delay necesario para la libreria y visualizacion de videos

        if key == 27:  # Si pulsamos ESC, se detiene el video (ESC corresponde al valor 27)
            break
    else:
        key = cv2.waitKey(30)  # Delay necesario para la libreria y visualizacion de videos

        if key == 27:  # Si pulsamos ESC, se detiene el video (ESC corresponde al valor 27)
            break

cap.release()
cv2.destroyAllWindows()

fin = time.time()
print("Tiempo de ejecución: ", fin - inicio)
