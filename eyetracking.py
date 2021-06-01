import numpy as np
import cv2

import time

capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)

    threshold = cv2.threshold(img_blurred, 50, 255, cv2.THRESH_BINARY_INV)[1]
    contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    for contour in contours:
        cv2.drawContours(img, [contour], -1, (0, 0, 255), 3)

    cv2.imshow("img", img)

    key_code = cv2.waitKey(30)
    if key_code == 27:
        break

cv2.destroyAllWindows()
