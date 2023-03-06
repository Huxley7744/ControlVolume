import cv2
import time
import numpy as np
import TrackHand as th
import math
# to use pycaw, begin
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# to use pycaw, end

# weight and height of the frame
wCam, hCam = 640, 480
# wCam, hCam = 1280, 720 # bigger one

# get webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = th.handDetector(detectionCon=0.7)

vol_per = 0
vol_bar = 400
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

while True:
    # get one frame
    ret, frame = cap.read()

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, personDraw=False)  # already draw it in findHands()
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])  # (thumb finger tip and index finger tip)'s position
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        # highlight the thumb and index finger tip
        cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        # draw a line between the thumb and index finger tip
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # calculate the distance between the thumb and index finger tip
        finger_length = math.hypot(x2 - x1, y2 - y1)
        # print(finger_length)
        # if the distance is less than 50, change the color to green
        if finger_length < 50:
            cv2.circle(frame, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # distance(15, 150) -> volume(-63, 0)
        # convert the distance to volume
        vol = np.interp(finger_length, [15, 150], [min_vol, max_vol])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        # convert the volume to volume bar
        vol_bar = np.interp(finger_length, [15, 150], [400, 150])
        # get the percentage of the volume
        vol_per = np.interp(finger_length, [15, 150], [0, 100])

    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, f'{int(vol_per)}%', (42, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS:{int(fps)}', (16, 84), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    # show the frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the webcam
cap.release()
cv2.destroyAllWindows()
