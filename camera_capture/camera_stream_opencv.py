import cv2
import numpy as np

class Camera:

    def __init__(self, cap_width=1920, cap_height=1080, disp_width=1280, disp_height=720, framerate = 30, flip_method = 2):
        self.cap_width = cap_width
        self.cap_height = cap_height
        self.disp_width = disp_width
        self.disp_height = disp_height
        self.framerate = framerate
        self.flip_method = flip_method
    def gstreamer_pipeline_show(self):
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

    def stream(self):
        print(gstreamer_pipeline_show(self))
        #opencv documentation for VideoCapture function, https://docs.opencv.org/4.1.1/d8/dfe/classcv_1_1VideoCapture.html#aabce0d83aa0da9af802455e8cf5fd181
        capture = cv2.VideoCapture(gstreamer_pipeline_show(self), cv2.CAP_GSTREAMER)
        #opencv documentation for VideoCapture object isOpened method, https://docs.opencv.org/4.1.1/d8/dfe/classcv_1_1VideoCapture.html#a9d2ca36789e7fcfe7a7be3b328038585
        if capture.isOpened():
            # opencv documentation for namedWindow, https://docs.opencv.org/4.1.1/d7/dfc/group__highgui.html#ga5afdf8410934fd099df85c75b2e0888b
            window = cv2.namedWindow("Arducam", cv2.WINDOW_AUTOSIZE)
            #getWindowProperty documentation: https://docs.opencv.org/4.1.1/d7/dfc/group__highgui.html#gaaf9504b8f9cf19024d9d44a14e461656
            while cv2.getWindowProperty("Arducam", 0) >= 0:
                ret, frame = capture.read()
                cv2.imshow("Arducam", frame)
                # This also acts as
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
            capture.release()
            cv2.destoryAllWindows()
        else:
            print("Unable to open capture stream")

if __name__ == "__main__":
cam = Camera()
    camera_show()