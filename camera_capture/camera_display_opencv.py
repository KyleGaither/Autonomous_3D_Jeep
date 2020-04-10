# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import time
import numpy as np

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

def HSV(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	hsv = cv2.GaussianBlur(hsv,(5,5),0)
	return hsv

#Jet-Jeep tutorial uses blue painters tape as lane lines. Extract only the bluish parts of the image and change them to white.
# convert rest of the colors to black to
def create_mask(hsv):
	lower_blue = np.array([60, 40, 40])
	upper_blue = np.array([150, 255, 255])
	mask = cv2.inRange(hsv, lower_blue, upper_blue)
	return mask
	
#Apply Canny edge detection to change all colors to black except for the edges of semi-continuous lines
def canny_edge(mask):
	edges = cv2.Canny(mask, 200, 400)
	return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (width*0.6, height * 1 / 2 ),
        (width*0.5, height * 1 / 2),
        (width*0.9, height),
        (width*0.15, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments

def gstreamer_pipeline(
    capture_width=720,
    capture_height=480,
    display_width=720,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=2))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    '''
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
        
    '''
    num_frames = 50
    time_start = time.perf_counter()
    for i in range(num_frames):
    	ret, frame = cap.read()
    	hsv = HSV(frame)
    	mask = create_mask(hsv)
    	edges = canny_edge(mask)
    	roi = region_of_interest(edges)
    	line_segments = detect_line_segments(roi)
    	
    time_end = time.perf_counter()
    seconds = time_end - time_start
    print("time taken: {0} second(s)".format(seconds))
    fps = num_frames/seconds
    print("est. fps: {0} frames/sec)".format(fps))
    cv2.imshow("arducam", roi)
    k = cv2.waitKey(0)
    if k == 27:
    	cap.release()
    	cv2.destroyAllWindows()
    	
    


if __name__ == "__main__":
	show_camera()
    
