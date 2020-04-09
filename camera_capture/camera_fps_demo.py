import camera_capture_opencv as Cam
import framesPerSecond
import cv2
import numpy as np
import threading
from time import sleep

max_frames = 200
disp_output = False
camera = Cam.Camera.start()
fps = framesPerSecond.FPS().start()

while fps._numFrames < max_frames:
    frame = camera.read()

    if disp_output:
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
camera.stop()







def start(self):
	    print(self.gstreamer_pipeline_show())
	    #opencv documentation for VideoCapture function, https://docs.opencv.org/4.1.1/d8/dfe/classcv_1_1VideoCapture.html#aabce0d83aa0da9af802455e8cf5fd181
	    capture = cv2.VideoCapture(self.gstreamer_pipeline_show(), cv2.CAP_GSTREAMER)
	    #opencv documentation for VideoCapture object isOpened method, https://docs.opencv.org/4.1.1/d8/dfe/classcv_1_1VideoCapture.html#a9d2ca36789e7fcfe7a7be3b328038585
	    if capture.isOpened():
	        # opencv documentation for namedWindow, https://docs.opencv.org/4.1.1/d7/dfc/group__highgui.html#ga5afdf8410934fd099df85c75b2e0888b
	        window = cv2.namedWindow("Arducam", cv2.WINDOW_AUTOSIZE)
	        #getWindowProperty documentation: https://docs.opencv.org/4.1.1/d7/dfc/group__highgui.html#gaaf9504b8f9cf19024d9d44a14e461656
	        while cv2.getWindowProperty("Arducam", 0) >= 0:
	            (ret, frame) = capture.read()
	            cv2.imshow("Arducam", frame)
	            # This also acts as
	            keyCode = cv2.waitKey(1) & 0xFF
	            # Stop the program on the ESC key
	            if keyCode == 27:
	                break
	        capture.release()
	        cv2.destoryAllWindows()
	    else:
	        print("Unable to open capture stream")


