import numpy as np
import cv2

gStreamer_pipeline = "nvarguscamerasrc ! "
					 "video/x-raw(memory:NVMM), "
        			 "width=(int)3280, height=(int)2464, "
        			 "format=(string)NV12, framerate=(fraction)21/1 ! "
        			 "nvvidconv flip-method=2 ! "
       			 	 "video/x-raw, width=1280, height=720 format=(string)BGRx ! "
        			 "videoconvert ! "
        			 "video/x-raw, format=(string)BGR ! appsink"
cap = cv2.VideoCapture(gStreamer_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
	print("Cannot open Camera")
	exit()
while True:
	#Capture frame by frame
	ret, frame = cap.read()
	
	# if frame is read correctly ret is True
	if not ret:
		print("Can't receive frame. Exiting.....")
		break
	#display the resulting frame 
	cv2.imshow("frame", frame)
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
