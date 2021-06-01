from pynput.mouse import Button, Controller
from screeninfo import get_monitors
import numpy as np
import cv2


from modules.handtracking import HandTracker 
import time

hand_tracker = HandTracker()
mouse = Controller()

camera_width, camera_height = 640, 480
frame_reduction = 100
smoothing = 7

previous_time = 0
current_time = 0

ploc_x, ploc_y = 0, 0
cloc_x, cloc_x = 0, 0

hand_tracker.create_window()

capture = cv2.VideoCapture(0)
capture.set(3, camera_width)
capture.set(4, camera_height)

monitor = get_monitors()[0]
screen_width, screen_height = monitor.width, monitor.height

while cv2.getWindowProperty(hand_tracker.window_name, 0) >= 0:
    success, img = hand_tracker.capture.read()

    current_time = time.time()
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time
    
    def finish_loop():
        flipped_img = cv2.flip(img, 1)
        cv2.putText(flipped_img, "fps: " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv2.imshow(hand_tracker.window_name, flipped_img)
        cv2.waitKey(50)

    hands = hand_tracker.find_hands(img, True)
    
    cv2.rectangle(img, (frame_reduction, frame_reduction), (camera_width - frame_reduction, camera_height - frame_reduction), (255, 0, 255), 2)
    
    if not len(hands) > 0:
        finish_loop()
        continue

    landmark_list, bounding_box = hand_tracker.find_position(img, hands[0])

    if not len(landmark_list) > 0:
        finish_loop()
        continue

    x1, y1 = landmark_list[8][1:]
    x2, y2 = landmark_list[12][1:]
    fingers = hand_tracker.fingers_up(landmark_list)

    if fingers[1] == 1 and fingers[2] == 0:
        x3 = np.interp(x1, (frame_reduction, camera_width - frame_reduction), (0, screen_width))
        y3 = np.interp(y1, (frame_reduction, camera_height - frame_reduction), (0, screen_height))

        cloc_x = ploc_x + (x3 - ploc_x) / smoothing
        cloc_y = ploc_y + (y3 - ploc_y) / smoothing
        
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        mouse.position = (screen_width - cloc_x, cloc_y)

        ploc_x, ploc_y = cloc_x, cloc_y

    if fingers[1] == 1 and fingers[2] == 1:
        length, line_info = hand_tracker.distance(landmark_list[8], landmark_list[12], img)

        if length < 40:
            cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
            mouse.press(Button.left)
            mouse.release(Button.left)

    finish_loop()
"""
    # 8. Both Index and middle fingers are up : Clicking Mode
    
    
    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
"""