import numpy as np
import cv2

eye_cascade = cv2.CascadeClassifier("cascade-eye.xml")
face_cascade = cv2.CascadeClassifier("cascade-face.xml")
capture = cv2.VideoCapture(0)

cache_faces = []
cache_eyes = []
cache_ex = 0

offset = 40

def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

while True:
    _, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    cv2.imshow("gray", gray)

    if not len(faces) > 0:
        faces = cache_faces

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w,y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if not len(eyes) > 0:
            eyes = cache_eyes

        if not len(eyes) > 0:
            continue

        ex_list = []
        for eye in eyes:
            ex_list.append(eye[0])

        find_ex = closest(ex_list, cache_ex)
        
        for ex, ey, ew, eh in eyes:
            if eye[0] == find_ex:
                eye_img = []

                try:
                    eye_img = roi_gray[ex-offset:ex+ew+offset, ey-offset:ey+eh+offset]
                except:
                    eye_img = roi_gray[ex:ex+ew, ey:ey+eh]

                img_blurred = cv2.GaussianBlur(eye_img, (5, 5), 0)
                threshold = cv2.threshold(img_blurred, 50, 255, cv2.THRESH_BINARY_INV)[1]
                contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

                for contour in contours:
                    cv2.drawContours(eye_img, [contour], -1, (0, 0, 255), 3)

                cv2.imshow("one eye", threshold)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cache_ex = ex

        cache_faces = faces
        cache_eyes = eyes

    cv2.imshow("eye detection", img)
    key_code = cv2.waitKey(30)
    if key_code == 27:
        break

capture.release()
cv2.destroyAllWindows()
