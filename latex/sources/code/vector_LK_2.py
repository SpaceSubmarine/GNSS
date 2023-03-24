import cv2
import numpy as np


# Calling video file or initializing webcam (for webcam input must be 0)
# video_path = './videos/videos_laboratorio/221214-Humo_V01.2.mp4'
video_path = './videos/videos_laboratorio/VID_20221230_073740_HSR_240.mp4'
cap = cv2.VideoCapture(video_path)
escalado = 1  # escalado de la imagen
if video_path == 0:
    escala = 1  # 4 / escalado # escala de las linias (vectores). Recomendados (2-10)
else:
    escala = 2 / escalado  # Escala de las linias (vectores). Recomendados (2-10)
m_dens = 5  # Para mayor definicion, valores menores. Recomendados(5-10)
Gray_umbral = 23  # para vr todos los puntos de velocidad (0), recomendado(25 o 30) para humo laboratorio
Gray_limit = True  # Para desactivar el limitador de la escala de grises
fondo_negro = False  # True para no ver el video debajo

# Definicion del color de los vectores
e_color_B = 0
e_color_G = 200
e_color_R = 255


def rescaleframe(frame, scale=escalado):  # Tener en cuenta "shadowing" y "global variable"
    width = int(frame.shape[1] * scale)
    heigh = int(frame.shape[0] * scale)
    dimensions = (width, heigh)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


# Creamos una ventana para mostrar la imagen
cv2.namedWindow("Optical Flow")

# Inicializamos el primer frame
_, prev_frame = cap.read()  # la primera variable (si lee el frame o no, la despreciamos con _)
prev_frame = rescaleframe(prev_frame)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# imagen negra
width = int(prev_gray.shape[1])
heigh = int(prev_gray.shape[0])
black_image = np.zeros((heigh, width, 3), np.uint8)

# Creamos el objeto de flujo óptico
flow = cv2.optflow.DualTVL1OpticalFlow_create()
N_frame = 0
while True:
    # Leemos el siguiente frame de la webcam
    _, curr_frame = cap.read()
    curr_frame = rescaleframe(curr_frame)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    is_flow = np.zeros(curr_gray.shape)
    if Gray_limit:
        for i in range(heigh):
            for j in range(width):
                if curr_gray[i, j] <= Gray_umbral:
                    curr_gray[i, j] = 0
                else:
                    is_flow[i, j] = 1
    # cv2.imshow("gray", is_flow)
    # Calculamos el flujo óptico entre el frame anterior y el actual
    flow_vectors = flow.calc(prev_gray, curr_gray, None)

    # Creamos una imagen en blanco y negro para dibujar los vectores de flujo
    flow_image = np.zeros_like(prev_frame)

    # Dibujamos los vectores de flujo sobre la imagen en blanco y negro
    for y in range(0, flow_image.shape[0], m_dens):
        for x in range(0, flow_image.shape[1], m_dens):
            fx, fy = flow_vectors[y, x]
            f = ((fx**2)+(fy**2))**(1/2)
            if is_flow[y, x] != 0:
                cv2.line(flow_image, (x, y), (x + int(fx * escala), y + int(fy * escala)),
                        (e_color_B, e_color_G, e_color_R), 1)

    # Superponemos la imagen con los vectores de flujo sobre la imagen actual (se aplica un 1 de transparencia)
    if fondo_negro:
        result_image = cv2.addWeighted(black_image, 1, flow_image, 1, 0)
    else:
        result_image = cv2.addWeighted(curr_frame, 0.5, flow_image, 1, 0)

    # Mostramos la imagen resultante
    cv2.imshow("Optical Flow", result_image)
    #cv2.imwrite('./Frame_generation/opencv/TVL1/' + str(N_frame) + '.png', result_image)
    N_frame += 1
    print(N_frame)
    # Actualizamos el frame anterior y esperamos a que se pulse una tecla
    prev_gray = curr_gray
    if cv2.waitKey(1) == 27:
        break

# Liberamos la webcam y cerramos la ventana
cap.release()
cv2.destroyAllWindows()
