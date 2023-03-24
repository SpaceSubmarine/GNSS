from pathlib import Path
import pandas as pd
import numpy as np
import math
import time
import glob
import cv2
import os

start = time.time()
# DESCRIPTION: This script performs object tracking.
# It is capable of storing data in CSV for each frame of the centroid positions
# for each pixel and generate DataFrames with object identifiers with their position, velocity
# and frame. This data must be exported for each frame, in this way, a script with
# matplotlib.pyplot can generate X and Y velocity matrices in order to plot those velocities
# for each time instant and thus generate a transient animation with opencv or matplotlib.

# Inputs
tol = 2  # Tolerance 0 with large videos (distance between the box edges and the contour)
print("HSV = Press 1 ")
print("GRAY = Press 0 ")
HSV = int(input())  # 200 is recommended for videos and 0 for the particle
print("Enter the minimum area in pixels:")
area_low = int(input())  # 200 is recommended for videos and 0 for the particle
print("Enter the maximum area in pixels:")
area_high = int(input())  # 100000
print("Enter the scale factor:")
escalado = float(input())  # Scale for video visualization
print("Enter the tracking distance in pixels:")
dist = int(input())  # escalado * 6 # distance in pixels
contours = True  # Visualization of contours
squares = True  # Visualization of boxes
step_by_step = False

print("Do you want to save databases?:")
print("YES = Press 1 ")
print("NO = Press 0 ")
save_data = int(input())
see_velocities = False
trackerCV = False  # Activate or deactivate the previously selected tracker
Show_ROI = False
Show_Frames = True
Show_ThreshHold = False
Show_MASK_conn = False
print("Do you want to save the generated frames?:")
print("YES = Press 1 ")
print("NO = Press 0 ")
Save_Frames = int(input())
Save_thresh = False
Save_conn = False
clean_frames = True
clean_data = True
cutout = False
noise_red = 50  # Noise reduction of the threshold (50: particle. Higher number, less noise)
connection_type = 4  # type of connectivity between components
hmin = 165  # 20 # 0
smin = 0  # 137 #0
vmin = 22  # 58 #255
hmax = 185  # 41 #90
smax = 252  # 255 #28
vmax = 255  # 255 #255
e = 0

# Colors
# to be able to adjust colors without having to do it for each putText
blue = (255, 0, 0)
red = (0, 0, 255)
orange = (0, 150, 255)

# path = "./videos/videos_laboratorio/particulas_slow.mp4"
path = './videos/videos_laboratorio/VID_20221230_073740_HSR_240.mp4'

# Data Cleaning
if clean_data:
    files = glob.glob('./data/DF_Values/')
    files1 = glob.glob('./data/DF_Labels/')
    for f in files:
        os.remove(f)  # Clean up the DF_Values folder
    for f in files1:
        os.remove(f)  # Clean up the DF_Labels folder
if clean_frames:
    files2 = glob.glob('./Frame_generation/opencv/Frames/*')
    files3 = glob.glob('./Frame_generation/opencv/Mask_thresh/*')
    files4 = glob.glob('./Frame_generation/opencv/Mask_Conn/*')
    for f in files2:
        os.remove(f)  # Clean up the opencv (Frames) folder
    for f in files3:
        os.remove(f)  # Clean up the opencv (Mask_thresh) folder
    for f in files4:
        os.remove(f)  # Clean up the opencv (Mask_Conn) folder


# We deende the function to rescale the image to better visualize the pixels
def rescaleframe(frame, scale=escalado):  # Take into account "shadowing" and "global variable"
    width = int(frame.shape[1] * scale)
    heigh = int(frame.shape[0] * scale)
    dimensions = (width, heigh)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


# Initializing frames
cap = cv2.VideoCapture(path)  # 0 si se quiere capturar webcam
thresh_function = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=noise_red, detectShadows=False)

# Initializations
df_centroids = []  # Initialization of centroids dataframe
N_frame = 0  # to know the frame
values_old = pd.DataFrame(np.zeros(50))  # initialization of previous values df
center_points_prev_frame = []  # initialization of previous frame centers
tracking_objects = {}  # initialization of tracking objects
track_id = 0  # initialization of object ID number
center_points = []  # initialize list of points

# Video visualization
while True:
    ret, frame_original = cap.read()
    # if testing with camera, flip the frame to see it mirrored
    if path == 0:
        frame_original = cv2.flip(frame_original, 1)
    if not ret:
        break

    # If cutout is selected
    if cutout:
        frame_original = frame_original[33:1400, 200:1809]
    frame_original = rescaleframe(frame_original)
    frame = frame_original.copy()

    # arrays with HSV minimum and maximum values
    lower_hsv = np.array([hmin, smin, vmin])
    upper_hsv = np.array([hmax, smax, vmax])

    # Convert BGR format to HSV (remember to convert back to BGR later)
    if HSV:
        hsv = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_BGR2HSV)
        MASK_Color = cv2.inRange(hsv, lower_hsv, upper_hsv)
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        MASK_Color = hsv

    height, width, _ = frame.shape  # height, width, _ = frame.shape     (with colors)

    # current frame point
    center_points_cur_frame = []

    # Object detection
    thresh = thresh_function.apply(MASK_Color)
    # cv2.imwrite('./testing/MASK_thresh/' + str(N_frame) + '.png', thresh)

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

    # Generation of dataframes per frame
    values = pd.DataFrame(values)  # This dataframe saves numeric values for each label
    filepath_values = Path('./datos/DF_Values/DF_Values_F' + str(N_frame) + '.csv')
    filepath_values.parent.mkdir(parents=True, exist_ok=True)
    values["Vx"] = np.nan  # initialization of the column of the dataframe per frame
    values["Vy"] = np.nan  # initialization of the column of the dataframe per frame
    values["VT"] = np.nan  # initialization of the column of the dataframe per frame
    label_ids = pd.DataFrame(label_ids)  # This dataframe is a matrix where the values are the labels
    filepath_labels = Path('.//data/DF_Labels/DF_Labels_F' + str(N_frame) + '.csv')
    filepath_labels.parent.mkdir(parents=True, exist_ok=True)

    # Region of interest within the video
    roi = frame[0:height, 0:width]  # It is the area in which the contours are desired to be plotted
    # The binary mask (thresh) is used to perform the 'Contours'
    detections = []
    MASK_BOX = frame
    ROI_number = 0
    N_boxes = 0
    cnts = cv2.enddContours(MASK_Conn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ROI = frame[y:y + h, x:x + w]

        # cv2.imwrite('./testing/MASK_BOX/' + str(N_frame) + '.png', MASK_BOX)
        ROI_number += 1
        if contours:  # If contours are activated, they are plotted:
            # Deendition of the contours for each accepted cluster
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)  # (type,(B, G, R), thickness)

        cx = int((x + x + w) / 2)  # x coordinate of the center of the box
        cy = int((y + y + h) / 2)  # y coordinate of the center of the box
        center_points_cur_frame.append((cx, cy))  # concatenate the new point
        # X,Y corner coordinates + width and height
        detections.append([x, y, w, h])
        # Deendition of labels for the squares in the video

        recuadro_actual = str(label_ids.loc[int(str(c[0][0][1])), int(str(c[0][0][0]))])
        if squares:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 210), 1)
            cv2.putText(roi, "X:" + str(cx) + " Y:" + str(cy), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 150, 255))

    # TRACKING
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
        # Iterate through all the objects being tracked
        for object_id, pt2 in tracking_objects_copy.items():
            object_exist = False
            # Iterate through all the center points detected in the current frame
            for pt in center_points_cur_frame_copy:
                distance = ((pt2[0] - pt[0]) ** 2 + (pt2[1] - pt[1]) ** 2) ** (1 / 2)
                # Calculate the velocity of the object
                VT = distance  # pixels per frame
                vx = pt2[0] - pt[0]  # pixels per frame
                vy = pt2[1] - pt[1]  # pixels per frame

                max_vt.append(int(VT))
                # If the distance is less than a threshold value, consider the object and center point as a match
                if distance < dist:
                    if pt in center_points_cur_frame:
                        tracking_objects[object_id] = pt
                        object_exist = True
                        center_points_cur_frame.remove(pt)
                        cv2.putText(frame, str(object_id), pt, cv2.FONT_HERSHEY_DUPLEX, 0.4, blue, 1)
                        cv2.circle(frame, pt, 1, (0, 170, 0), -1)
                        dfValues.append([object_id, pt2[0], pt2[1], vx, vy, VT])
                        if see_velocities:
                            VT = "V: " + str("%.2f" % round(VT, 2))
                            Vx = "Vx: " + str(pt2[0] - pt[0])
                            Vy = "Vy: " + str(pt2[1] - pt[1])
                            cv2.putText(frame, VT, (pt2[0], pt2[1] + 10), cv2.FONT_HERSHEY_DUPLEX, 0.35, red, 1)
                            cv2.putText(frame, (Vx + " " + Vy), (pt2[0], pt2[1] + 25), cv2.FONT_HERSHEY_DUPLEX, 0.35,
                                        red, 1)
                    continue

            # Remove lost IDs
            if not object_exist:
                tracking_objects.pop(object_id)

        # Add new IDs
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    N_frame += 1
    dfValues = pd.DataFrame(dfValues, columns=['ID', 'Px', 'Py', 'Vx', 'Vy', 'VT'])
    # Saving the databases
    if save_data:
        label_ids.to_csv(filepath_labels)
        dfValues.to_csv(filepath_values)

    # values_old = values
    # endding the minimum and maximum values of VT
    min_vt = np.min(np.array(max_vt))
    max_vt = np.max(np.array(max_vt))
    median_vt = (max_vt - min_vt) / 2

    # Informational texts for each frame
    cv2.putText(frame, "AD Tracker", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
    cv2.putText(frame, "Clusters : " + str(int(totalLabels)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, orange, 2)
    cv2.putText(frame, "Frame : " + str(N_frame), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, orange, 2)
    cv2.putText(frame, "Sacle : " + str(escalado), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, orange, 2)

    # Visualizations during execution
    if Show_ROI:
        cv2.imshow("Roi", roi)  # To visualize the video of only the region of interest
    if Show_Frames:
        cv2.imshow("TRACKING", frame)  # To visualize the video with contours and squares
    if Show_ThreshHold:
        cv2.imshow("ThresHolding IMAGE", thresh)  # To visualize the video of ThresHolding
    if Show_MASK_conn:
        cv2.imshow("MASK_Conn IMAGE", MASK_Conn)  # To visualize the video of the ThresHolding
    if Save_Frames:
        cv2.imwrite('./Frame_generation/opencv/Frames/' + str(N_frame) + '.png', frame)
    if Save_thresh:
        cv2.imwrite('./Frame_generation/opencv/Mask_thresh/' + str(N_frame) + '.png', thresh)
    if Save_conn:
        cv2.imwrite('./Frame_generation/opencv/Mask_Conn/' + str(N_frame) + '.png', MASK_Conn)
    print(N_frame)

    # Copying the points
    center_points_prev_frame = center_points_cur_frame.copy()

    # Choosing to visualize the normal video or pass frames manually
    if step_by_step:
        key = cv2.waitKey(0)  # Delay necessary for the library and video visualization

        if key == 27:  # If we press ESC, the video stops (ESC corresponds to the value 27)
            break
    else:
        key = cv2.waitKey(30)  # Delay necessary for the library and video visualization

        if key == 27:  # If we press ESC, the video stops (ESC corresponds to the value 27)
            break

cap.release()
cv2.destroyAllWindows()

end = time.time()
print("Tiempo de ejecución: ", end - start)
