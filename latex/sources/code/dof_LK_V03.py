import numpy as np
import time
import cv2

# More info:
# https://learnopencv.com/optical-flow-in-opencv/#dense-optical-flow-lk

start = time.time()
video_path = './videos/videos_laboratorio/VID_20221202_125801_HSR_240031609.mp4'
scaling = 0.5
method = cv2.optflow.calcOpticalFlowSparseToDense
gray_threshold = 25
Vb_filter = 255
gray_limit = True
save_frames = False

# Values to consider as parameters in the Lucas-Kanade method (empirical):
# flow = method(old_frame, new_frame, None, *params)
# flow = method(old_frame, new_frame, None, 60, 4, 0.0009, True, 10000000, 0.1)
# flow = method(old_frame, new_frame, None, 8, 512, 0.01, True, 1500, 1)
# flow = method(old_frame, new_frame, None, 8, 512, 0.01, True, 1500, 1)
# flow = method(old_frame, new_frame, None, 8, 128, 0.05, True, 10, 0.5)

# Standarized Parameters for Lucas Kande optical flow
# lk_params = dict[None, 8, 128, 0.05, True, 500, 1.5]


# Deende the function to rescale the image to see pixels better
def rescaleframe(frame, scale=scaling):  # Keep in mind "shadowing" and
    width = int(frame.shape[1] * scale)
    heigh = int(frame.shape[0] * scale)
    dimensions = (width, heigh)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


# Read the first frame of the video
cap = cv2.VideoCapture(video_path)
ret, old_frame = cap.read()  # although the variable ret (bool) is saved, it will not be used

# Rescale the video frame
old_frame = rescaleframe(old_frame)

# Create HSV
hsv = np.zeros_like(old_frame)  # initialization of frame in HSV
hsv[..., 1] = 255

# Apply grayscale to the first frame of the video
old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

N_frame = 0
while True:
    # Read the next frame
    ret, new_frame = cap.read()

    frame_copy = new_frame
    # new_frame = cv2.flip(frame_copy, 1)  # Mirror for webcam

    if not ret:
        break

    # Preprocessing for exact method
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    width = new_frame.shape[1]
    height = new_frame.shape[0]

    new_frame = rescaleframe(new_frame)
    width_2 = new_frame.shape[1]
    height_2 = new_frame.shape[0]
    is_flow = np.zeros(new_frame.shape)
    if gray_limit:
        for i in range(height_2):
            for j in range(width_2):
                if new_frame[i, j] <= gray_threshold:
                    new_frame[i, j] = 0
                else:
                    is_flow[i, j] = 1

    #  Filter the grayscale with threshold
    new_frame = new_frame

    # Calculate Optical Flow

    flow = method(old_frame, new_frame, None, 8, 128, 0.05, True, 500, 0.5)
    # Encoding: convert the algorithm's output into Polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Use Hue and Value to encode the Optical Flow

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, Vb_filter, 255, cv2.NORM_MINMAX)
    # Convert HSV image into BGR for demo
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow("frame", frame_copy)

    bgr[:, :, 0] = bgr[:, :, 0] * is_flow
    bgr[:, :, 1] = bgr[:, :, 1] * is_flow
    bgr[:, :, 2] = bgr[:, :, 2] * is_flow
    cv2.imshow("optical flow", bgr)
    if save_frames:
        cv2.imwrite('./Frame_generation/opencv/LK2/' + str(N_frame) + '.png', bgr)

    print("LK_2", N_frame)
    N_frame += 1
    k = cv2.waitKey(25) & 0xFF

    if k == 27:
        break
    # Update the previous frame
    old_frame = new_frame
end = time.time()
print("Computational time: ", end - start)
