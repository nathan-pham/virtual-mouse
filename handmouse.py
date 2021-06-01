from pynput.mouse import Button, Controller
from screeninfo import get_monitors
import numpy as np
import cv2


from modules.handtracking import HandTracker 
import time

camera_width, camera_height = 640, 480
frame_reduction = 100
smoothing = 7

previous_time = 0
current_time = 0

plocX, plocY = 0, 0
clocX, clocY = 0, 0

capture = cv2.VideoCapture(0)
capture.set(3, camera_width)
capture.set(4, camera_height)

hand_tracker = HandTracker()

monitor = get_monitors()[0]
screen_width, screen_height = monitor.width, monitor.height

hand_tracker.create_window()

while cv2.getWindowProperty(hand_tracker.window_name, 0) >= 0:
    success, img = hand_tracker.capture.read()
    
    current_time = time.time()
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time
    
    hands = hand_tracker.find_hands(img, True, fps)
    
    if len(hands) > 0:
        landmark_list, bounding_box = hand_tracker.find_position(img, hands[0])

    cv2.imshow(hand_tracker.window_name, img)
    cv2.waitKey(50)

"""
while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)
    
    # 3. Check which fingers are up
    fingers = detector.fingersUp()
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
    (255, 0, 255), 2)
    # 4. Only Index Finger : Moving Mode
    if fingers[1] == 1 and fingers[2] == 0:
        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
    
        # 7. Move Mouse
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY
        
    # 8. Both Index and middle fingers are up : Clicking Mode
    if fingers[1] == 1 and fingers[2] == 1:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(8, 12, img)
        print(length)
        # 10. Click mouse if distance short
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
            15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()
    
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