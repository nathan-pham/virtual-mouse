import mediapipe as mp
import numpy as np
import cv2

import time
import math

class HandTracker():
    def __init__(self, window_name="hand tracking", mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.tip_ids = [4, 8, 12, 16, 20]
        self.window_name = window_name

        self.model = mp.solutions.hands
        self.draw = mp.solutions.drawing_utils
        self.hands = self.model.Hands(mode, max_hands, detection_confidence, track_confidence)

    def create_window(self):
        self.capture = cv2.VideoCapture(0)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.startWindowThread()

    def find_hands(self, img, draw=True):
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        landmarks = results.multi_hand_landmarks or []

        if draw:
            for landmark in landmarks:
                self.draw.draw_landmarks(img, landmark, self.model.HAND_CONNECTIONS)

        return landmarks
    
    def find_position(self, img, hand, draw=True):
        height, width = img.shape[:2]
        x_list = []
        y_list = []
        landmarks = []
        bounding_box = [0, 0, 0, 0]

        for id, landmark in enumerate(hand.landmark):
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            x_list.append(cx)
            y_list.append(cy)
            landmarks.append([id, cx, cy])

            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        xmin, xmax = min(x_list), max(x_list)
        ymin, ymax = min(y_list), max(y_list)
        bounding_box = xmin, ymin, xmax, ymax

        if draw:
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        
        return landmarks, bounding_box
    
    def fingers_up(self, landmarks):
        fingers = [1 if landmarks[self.tip_ids[0]][1] > landmarks[self.tip_ids[0] - 1][1] else 0]
        for id in range(1, 5):
            fingers.append(1 if landmarks[self.tip_ids[id]][2] < landmarks[self.tip_ids[id] - 2][2] else 0)

        return fingers

    def distance(self, p1, p2, img, draw=True):
        x1, y1 = p1[1:]
        x2, y2 = p2[1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        
        return length, [x1, y1, x2, y2, cx, cy]

def main():
    hand_tracker = HandTracker()
    previous_time = 0
    current_time = 0

    hand_tracker.create_window()

    while cv2.getWindowProperty(hand_tracker.window_name, 0) >= 0:
        success, img = hand_tracker.capture.read()

        current_time = time.time()
        fps = int(1 / (current_time - previous_time))
        previous_time = current_time
        cv2.putText(img, "fps: " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

        hands = hand_tracker.find_hands(img, True)
        
        if len(hands) > 0:
            landmark_list, bounding_box = hand_tracker.find_position(img, hands[0])

        cv2.imshow(hand_tracker.window_name, img)
        cv2.waitKey(50)


if __name__ == "__main__":
    main()
