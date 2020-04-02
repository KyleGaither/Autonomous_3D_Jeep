import cv2

cv2_pipeline ="nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink" 

print(cv2_pipeline) 


cap = cv2.VideoCapture(cv2_pipeline, cv2.CAP_GSTREAMER)
if cap.isOpened():
	window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
	# Window
	while cv2.getWindowProperty("CSI Camera", 0) >= 0:
		ret_val, img = cap.read()
		cv2.imshow("CSI Camera",img)
		# This also acts as
		keyCode = cv2.waitKey(30) & 0xFF
		# Stop the program on the ESC key
		if keyCode == 27:
			break
	cap.release()
	cv2.destroyAllWindows()
else:
	print("Unable to open camera")

