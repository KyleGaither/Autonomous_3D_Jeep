#function from jetsonhacks simplecamera.py, github_repo: https://github.com/JetsonHacksNano/CSI-Camera.git
import cv2
import numpy as np
def gstreamer_pipeline_show(
    cap_width=1280, 
    cap_height=720, 
    disp_width=1280, 
    disp_height=720,
    framerate = 30,
    flip_method = 2,
):
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
            cap_width,
            cap_height,
            framerate,
            flip_method,
            disp_width,
            disp_height,
        )
    )

def camera_show():
    print(gstreamer_pipeline_show())
    #opencv documentation for VideoCapture function, https://docs.opencv.org/4.1.1/d8/dfe/classcv_1_1VideoCapture.html#aabce0d83aa0da9af802455e8cf5fd181
    capture = cv2.VideoCapture(gstreamer_pipeline_show(), cv2.CAP_GSTREAMER)
    #opencv documentation for VideoCapture object isOpened method, https://docs.opencv.org/4.1.1/d8/dfe/classcv_1_1VideoCapture.html#a9d2ca36789e7fcfe7a7be3b328038585
    if capture.isOpened():
        # opencv documentation for namedWindow, https://docs.opencv.org/4.1.1/d7/dfc/group__highgui.html#ga5afdf8410934fd099df85c75b2e0888b
        window = cv2.namedWindow("Arducam", cv2.WINDOW_AUTOSIZE)
        #getWindowProperty documentation: https://docs.opencv.org/4.1.1/d7/dfc/group__highgui.html#gaaf9504b8f9cf19024d9d44a14e461656
        while cv2.getWindowProperty("Arducam", 0) >= 0:
            ret, frame = capture.read()
            edges = find_edges(frame)
            roi = region_of_interest(edges)

            cv2.imshow("Arducam", roi)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        capture.release()
        cv2.destoryAllWindows()
    else:
        print("Unable to open capture stream")

def find_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60, 40, 40])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges