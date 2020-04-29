import math
import numpy as np
import logging
import time
import cv2
from adafruit_servokit import ServoKit  #library for controlling pwm board
import libJeep as Jeep

Cam = Jeep.Camera()
motor = Jeep.Motor(0.12, 0, 1)
cap = Cam.stream()

if cap.isOpened():
	# Window
	current_steering_angle = 90
	for i in range(420):
	    ret_val, img = cap.read()
	    lane_lines = Jeep.detect_lane(img)
	    new_steering_angle = Jeep.compute_steering_angle(img,lane_lines)
	    stabilized_angle = Jeep.stabilize_steering_angle(current_steering_angle, new_steering_angle,len(lane_lines))
	    motor.steer(stabilized_angle)
	    motor.speed(0.1, stabilized_angle)
	    current_steering_angle = stabilized_angle
else:
	print("Unable to open camera")

motor.speed(0,350)
motor.steer(100)

