import cv2
import numpy as np
import threading as Thread

class Camera:

	def __init__(self, cap_width=1280, cap_height=720, disp_width=720, disp_height=480, framerate = 30, flip_method = 2):
		#initialize arguments for gstreamer pipeline
	    self.cap_width = cap_width
	    self.cap_height = cap_height
	    self.disp_width = disp_width
	    self.disp_height = disp_height
	    self.framerate = framerate
	    self.flip_method = flip_method

		#initialize camera stream and capture initial image
		self.capture = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
		(self.ret, self.frame) = self.capture.read()

		#initialize variable to determine if thread should be stopped
		self.stopped = False

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
	
	def start(self):
		Thread(target=self.update, args=()).start()
		return self
	
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.ret, self.frame) = self.capture.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True