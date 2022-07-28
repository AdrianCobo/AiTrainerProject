import cv2
import numpy as np
import time
import PoseEstimationModule as pm

cap = cv2.VideoCapture("Media/bicepsCurl.mp4")

pTime = 0
detector = pm.poseDetector()
count = 0
dir = 0  # 0 = up, 1 = down

while cap.isOpened():
    success, img = cap.read()  # comment if you want to use a image
    # img = cv2.imread("Media/fondosEnParalelas.jpeg") uncomment if you want to use a image

    # if frame is read correctly ret is True
    if not success:  # comment if you want to use a image
        print("Can't receive frame (stream end?). Exiting ...")
        break

    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    if len(lmList) != 0:
        # Right Arm
        # detector.findAngle(img, 12, 14, 16)
        # Left Arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        # 650 and 100 = coordinates in pixels for the height of the bar
        bar = np.interp(angle, (220, 310), (650, 100))
        # check for the dumbbell curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        elif per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        # Draw curl
        cv2.rectangle(img, (0, 450), (250, 720), color, cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670),
                    cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        # Draw Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color,  3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650),
                      color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75),
                    cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    # exit pressing q
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
