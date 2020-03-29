import numpy as np
import cv2

cap = cv2.VideoCapture(1)
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
	cv2.imshow('frame')
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
