import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np

cap = cv2.VideoCapture(0)

det = FaceDetector()
servoPos = [90, 90]

while True:
    _, img = cap.read()
    img, bbox = det.findFaces(img, draw = False)
    if bbox:
        fx, fy = bbox[0]['center'][0], bbox[0]['center'][1]
        pos = [fx, fy]
        servoX = np.interp(fx, [0, 1280], [0, 180])
        servoY = np.interp(fy, [0, 720], [0, 180])

        if servoX < 0:
            servoX = 0
        elif servoX > 180:
            servoX = 180
        if servoY < 0:
            servoY = 0
        elif servoY > 180:
            servoY = 180
        
        servoPos[0] = servoX
        servoPos[1] = servoY

        cv2.circle(img, (fx, fy - 80), 80, (0, 0, 255), 2)
        cv2.putText(img, str(pos), (fx+15, fy-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2 )
        cv2.line(img, (0, fy - 80), (1280, fy), (0, 0, 0), 2)  # x line
        cv2.line(img, (fx, 720), (fx, 0), (0, 0, 0), 2)  # y line
        cv2.circle(img, (fx, fy - 80), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3 )

    else:
        cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (0, 360), (1280, 360), (0, 0, 0), 2)  # x line
        cv2.line(img, (640, 720), (640, 0), (0, 0, 0), 2)  # y line

    cv2.imshow("Headshot Predictor", img)
    if cv2.waitKey(1) == ord('q'):
        break
