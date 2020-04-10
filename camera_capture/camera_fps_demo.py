# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import time
import numpy as np
import threading


class VideoCaptureAsync:
	def __init__(self):
		self.cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
		self.grabbed, self.frame = self.cap.read()
		self.started = False
		self.read_lock = threading.Lock()
	
	def gstreamer_pipeline(self):
		return (
		    "nvarguscamerasrc ! "
		    "video/x-raw(memory:NVMM), "
		    "width=(int)1280, height=(int)720, "
		    "format=(string)NV12, framerate=(fraction)25/1 ! "
		    "nvvidconv flip-method=2 ! "
		    "video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! "
		    "videoconvert ! "
		    "video/x-raw, format=(string)BGR ! appsink"
		)

	def start(self):
		if self.started:
			print("Async video already started")
			return None
		self.started = True
		self.thread = threading.Thread(target=self.update, args=())
		self.thread.start()
		return self
	
	def update(self):
		while self.started:
			grabbed, frame = self.cap.read()
			with self.read_lock:
				self.grabbed = grabbed
				self.frame = frame
	
	def read(self):
		with self.read_lock:
			frame = self.frame.copy()
			grabbed = self.grabbed
		return grabbed, frame
	
	def stop(self):
		self.started = False
		self.thread.join()
	
	def __exit__(self, exec_type, exc_value, traceback):
		self.cap.release()
		

def test(n_frames = 500):
	cap = VideoCaptureAsync()
	cap.start()
	t0 = time.time()
	for i in range(n_frames):
		_, frame = cap.read()
		cv2.imshow("Arducam", frame)
		cv2.waitKey(1) & 0xFF
	
	print(f'FPS: {round(n_frames/(time.time()-t0), 2)}')	
	cap.stop()
	cv2.destroyAllWindows()
	
if __name__ == "__main__":
	test(1000)
	
