import cv2
import numpy as np


# Inputs:
color_map = True
blur_value = 4

# Abre la cámara
cap = cv2.VideoCapture('./videos/videos_laboratorio/221214-Humo_V01.3.mp4')
N_frame = 0
while True:
    # Lee el siguiente frame de la cámara
    _, frame = cap.read()
    # frame = mirrored_frame = cv2.flip(frame, 1)  # Mirror para webcam
    #frame = cv2.GaussianBlur(frame, (blur_value, blur_value), 0)
    # Convierte el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if N_frame > 0:
        # Calcular el flujo óptico utilizando el método Farnerback
        #flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.1, 9, blur_value, 3, 5, 1.2, 0)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.1, 9, blur_value, 3, 5, 1.2, 2)

        if color_map:
            ####################################################################################
            # Dibujar las líneas del flujo óptico en el frame actual
            hsv = np.zeros_like(prev_frame)
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = 255#ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # Muestra el frame en escala de grises
            cv2.imshow('Webcam', bgr)
            #cv2.imwrite('./Frame_generation/opencv/fnb/' + str(N_frame) + '.png', bgr)
    prev_gray = gray.copy()
    prev_frame = frame.copy()
    print(N_frame)
    N_frame += 1

    # Salir si se presiona 'q'
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    key = cv2.waitKey(1)  # Delay necesario para la libreria y visualizacion de videos

    if key == 27:  # Si pulsamos ESC, se detiene el video (ESC corresponde al valor 27)
        break
cap.release()
cv2.destroyAllWindows()


