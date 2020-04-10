import cv2
import numpy as np
import time

class Camera:

	def __init__(self, cap_width=1280, cap_height=720, disp_width=720, disp_height=480, framerate = 30, flip_method = 2):
		#initialize arguments for gstreamer pipeline
		self.cap_width = cap_width
		self.cap_height = cap_height
		self.disp_width = disp_width
		self.disp_height = disp_height
		self.framerate = framerate
		self.flip_method = flip_method

	def gstreamer_pipeline(self):
		return(
			"nvarguscamerasrc ! "
			"video/x-raw(memory:NVMM), "
			"width=(int)%d, height=(int)%d, "
			"format=(string)NV12, framerate=(fraction)%d/1 ! "
			"nvvidconv flip-method=%d ! "
			"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
			"videoconvert ! "
			"video/x-raw, format=(string)BGR ! appsink"
			% (
			self.cap_width,
			self.cap_height,
			self.framerate,
			self.flip_method,
			self.disp_width,
			self.disp_height,
			)
		)


	def start_camera(self):
		print(self.gstreamer_pipeline)
		return cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)

