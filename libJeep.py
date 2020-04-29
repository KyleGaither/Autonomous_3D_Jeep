import math
import numpy as np
import logging
import time
import cv2
from adafruit_servokit import ServoKit  #library for controlling pwm board


##############################  CAMERA CLASS  ##############################

class Camera:
	def __init__(self, dev=1, width=1280, height=720):
		self.dev = dev
		self.width = width
		self.height = height
		
	def stream(self):
		# On versions of L4T previous to L4T 28.1, flip-method=2
		# Use Jetson onboard camera
		#gst_str = "v4l2src ! video/x-raw, width=(int)1280, height=(int)720 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
		#return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
		return cv2.VideoCapture(0)

	def record(self, framerate):
		return cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*"MP4V"), framerate, (1280,720))


##############################  CAMERA CLASS END  ##############################


##############################  MOTOR CLASS  ##############################

class Motor:
	def __init__(self, max_speed, servo_index, esc_index):
		self.max_speed = max_speed
		self.servo_index = servo_index
		self.esc_index = esc_index
		self.kit = ServoKit(channels = 16)
		self.steer(90)
		self.speed(0,90)
	def steer(self, steering_angle):
		if steering_angle >= 130:
			self.kit.servo[self.servo_index].angle = 160
		else:
			self.kit.servo[self.servo_index].angle = steering_angle

	def speed(self, speed, steering_angle):
		if steering_angle < 120 and steering_angle > 70:
			self.kit.continuous_servo[self.esc_index].throttle = self.max_speed
		elif steering_angle >  300:
			self.kit.continuous_servo[self.esc_index].throttle = 0
		else:
			self.kit.continuous_servo[self.esc_index].throttle = speed

##############################  MOTOR CLASS END  ##############################
def HSV(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	return hsv

def detect_lane(frame):
    
    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)
    
    return lane_lines

def detect_edges(frame):
	hsv = HSV(frame)
	mask = create_mask(hsv)
	edges = canny_edge(mask)
	return edges

def create_mask(hsv):
	lower_blue = np.array([80, 40, 40])
	upper_blue = np.array([140, 255, 255])
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
        (0, height * 1/2 ),
        (width, height * 1/2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 15  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=10, maxLineGap=4)

    return line_segments

def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def compute_steering_angle(frame, lane_lines):
    """ Find the steering angle based on lane line coordinate
        We assume that camera is calibrated to point to dead center
    """
    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        return 90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    logging.debug('new steering angle: %s' % steering_angle)
    if  steering_angle < 50:
        steering_angle = 50

    if steering_angle > 140:
       steering_angle = 160
    return steering_angle


def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=15, 
			     max_angle_deviation_one_lane=10):
    """
    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """
    if num_of_lane_lines == 2 : 
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    logging.info('Proposed angle: %s, stabilized angle: %s' % (new_steering_angle, stabilized_steering_angle))
    return stabilized_steering_angle
