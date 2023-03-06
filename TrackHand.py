import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxhands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        """
        :param mode: 是否输入静态图像
        :param maxhands: 检测到手的最大数量
        :param complexity: 模型复杂度
        :param detectionCon: 检测手的置信度
        :param trackCon: 追踪手的置信度
        """
        self.mode = mode
        self.maxhands = maxhands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxhands, self.complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """
        :param img:要识别的一帧图像
        :param draw:是否对手的标志点进行绘图
        :return:绘画完成的一帧图像
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, personDraw=True):
        """
        :param img: 要识别的一帧图像
        :param handNo: 手的编号
        :param personDraw: 是否对手的标志点进行个性化绘图
        :return: 手的21个标志点位置
        """
        lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if personDraw:
                    cv2.circle(img, (cx, cy), int(w / 50), (255, 0, 255), cv2.FILLED)
        return lmList
