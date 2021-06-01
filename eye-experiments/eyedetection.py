import numpy as np
import cv2

eye_cascade = cv2.CascadeClassifier("cascade-eye.xml")
face_cascade = cv2.CascadeClassifier("cascade-face.xml")
capture = cv2.VideoCapture(0)

while True:
    _, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    cv2.imshow("gray", gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w,y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) > 0:
            ex, ey, ew, eh = eyes[0]
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.imshow("one eye", roi_gray[ex:ex+ew, ey:ey+eh])

    cv2.imshow("eye detection", img)
    key_code = cv2.waitKey(30)
    if key_code == 27:
        break

capture.release()
cv2.destroyAllWindows()
