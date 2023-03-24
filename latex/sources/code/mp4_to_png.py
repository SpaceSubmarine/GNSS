import cv2
vidcap = cv2.VideoCapture('./videos/videos_laboratorio/221213-Lab_s_V05.4.mp4')
save_path = "./videos/videos_laboratorio/frames/"
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(save_path + str(count) + ".png", image)
    success, image = vidcap.read()
    #print('Read a new frame: ', success)
    print(count)
    count += 1