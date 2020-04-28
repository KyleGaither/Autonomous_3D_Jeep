import math
import numpy as np
import logging
import time
import cv2
from adafruit_servokit import ServoKit  #library for controlling pwm board
import libJeep as Jeep

Cam = Jeep.Camera()
motor = Jeep.Motor(0.2, 0, 1)
cap = Cam.stream()
out = Cam.record(10)


while cap.isOpened():
	ret_val, img = cap.read()
	lane_line = Jeep.detect_lane(img)
	lane_img = Jeep.display_lines(img, lane_line)
	out.write(lane_img)
	# This also acts as
	keyCode = cv2.waitKey(30) & 0xFF
	# Stop the program on the ESC key
	if keyCode == 27:
		break
cap.release()
cv2.destroyAllWindows()
motor.steer(90)
motor.speed(0)



