import cv2
import numpy as np

#convert image from BGR colorspace to HSV and blur the image to reduce processing cost
def HSV(frame):
	blur = cv2.GaussianBlur(frame,(5,5),0)
	hsv = cv2.cv2Color(blur, cv2.COLOR_BGR2HSV)
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

#combine the 3 functions above to extract all edges from the image
def edge_detection(frame):
	hsv = HSV(frame)
	mask = create_mask(hsv)
	edges= canny_edge(mask)
	return edges

