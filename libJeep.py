import math
import numpy as np
import logging
import time
import cv2
from adafruit_servokit import ServoKit  #library for controlling pwm board

class Jeep:
    
    def __init__(
        self,
        # choose number of frames for jeep to run then stop. Meant for testing. set to 0 to run infinitely and use ^c to stop program
        num_frames,
        max_speed,
        # max deviation for stabilized steering angle function when 2 lanes are detected
        max_angle_deviation_2l,
        # max deviation for stabilized steering angle function when 1 lane is detected
        max_angle_deviation_1l,
        # chose to record video 
        record = False,
        #chose framerate depending on hardware computing power.
        #for Jetson nano: 24-30, Rpi: 15-24, jetson tx2: 30 - 40
        framerate = 24,
        #chose orientation. if upside down set to 0
        flip_method = 2,
        #select resolution for camera capture. reduce for higher fps
        width = 480,
        height = 320,
    ):
        self.num_frames = num_frames
        self.max_speed = max_speed
        self.max_angle_deviation_2l = max_angle_deviation_2l
        self.max_angle_deviation_1l = max_angle_deviation_1l
        self.record_flag = record

        #initilize motor control
        self.kit = ServoKit(channels=16)
        self.kit.servo[0].angle = 95
        self.kit.continuous_servo[1].throttle = 0

        #initilize camera
        if self.record_flag:
            self.cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_STREAMER)
            self.out = cv2.VideoWriter('output.avi', 'XVID', 20.0, (480, 320))
        else:
            self.cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_STREAMER)
            


    def gstreamer_pipeline(self):
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
                self.width,
                self.height,
                self.framerate,
                self.flip_method,
                self.width,
                self.height,
            )
        )

    def run(self):
        if self.record_flag:
            time_start = time.time()
            for i in range(self.num_frames):
                ret, frame = self.cap.read()
                lanes = self.detect_lane(frame)
                lane_img = self.display_lines(frame, lanes)
            cv2.imshow("arducam", lane_img)
            cv2.waitKey(0)
            self.cap.release()
            cv2.destroyAllWindows()


    def HSV(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv,(5,5),0)
        return hsv
    
    def create_mask(self, hsv):
        lower_blue = np.array([90, 40, 40])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        return mask

    def canny_edge(self, mask):
        edges = cv2.Canny(mask, 200, 400)
        return edges
    
    def detect_edges(self, frame):
        hsv = self.HSV(frame)
        mask = self.create_mask(hsv)
        edges = self.canny_edge(mask)
        return edges
    
    def region_of_interest(self, edges):
        height, width = edges.shape
        mask = np.zeros_like(edges)

        # only focus bottom half of the screen
        polygon = np.array([[
            (width*0.4, height * 1 / 2 ),
            (width*0.6, height * 1 / 2),
            (width, height),
            (0, height),
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        return cropped_edges
    
    def detect_line_segments(self, cropped_edges):
        # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
        rho = 1  # distance precision in pixel, i.e. 1 pixel
        angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
        min_threshold = 10  # minimal of votes
        line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                        np.array([]), minLineLength=8, maxLineGap=4)

        return line_segments

    def average_slope_intercept(self, frame, line_segments):
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
            lane_lines.append(self.make_points(frame, left_fit_average))

        right_fit_average = np.average(right_fit, axis=0)
        if len(right_fit) > 0:
            lane_lines.append(self.make_points(frame, right_fit_average))

        logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

        return lane_lines

    def make_points(self, frame, line):
        height, width, _ = frame.shape
        slope, intercept = line
        y1 = height  # bottom of the frame
        y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

        # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        return [[x1, y1, x2, y2]]

    def detect_lane(self, frame):
    
        edges = self.canny_edge(frame)
        cropped_edges = self.region_of_interest(edges)
        line_segments = self.detect_line_segments(cropped_edges)
        lane_lines = self.average_slope_intercept(frame, line_segments)
        
        return lane_lines

    def display_lines(self, frame, lines, line_color=(0, 255, 0), line_width=2):
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
        line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        return line_image

