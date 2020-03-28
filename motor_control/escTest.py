'''
This code is designed to test the servo motor and the esc controller simultaneously. Using the adafruit Circuit Python 
adafruit_servokit library
'''
from adafruit_servokit import ServoKit
import time
import sys
# Set up i2c connection between nano and PCA9685 pwm board
kit = ServoKit(channels=16)
inputAvailable = True

print("INSTRUCTIONS: This program is designed to test each motor individually until the user ends the program.\n"
"Select servo motor using keyword SERVO\n"
"Select drive motor using keyword ESC.\n"
"End program using keyword END.\n")

while inputAvailable:
    motorSelect = input("select which motor to test:\n")
    if motorSelect.upper() == "SERVO":
        servo_angle = int(input("Servo motor Selected, choose an angle: \n"))
        #choose index in servo method according to the index the servo is connected to on pwm board
        kit.servo[0].angle = servo_angle

    if motorSelect.upper() == "ESC":
        drive_speed = int(input("Drive motor selected, choose speed as a decimal between -1 to 1: \n"))
        kit.continuous_servo[1].throttle = drive_speed
        time.sleep(3)
        kit.continuous_servo[1].throttle = 0
    
    else:
        inputAvailable = False
        sys.exit("User ended motor test")

