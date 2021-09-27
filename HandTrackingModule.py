import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxhands=2, min_detection_con=0.5, min_tracking_con=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.min_detection_con = min_detection_con
        self.min_tracking_con = min_tracking_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxhands, self.min_detection_con, self.min_tracking_con)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 0), cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        lmList = detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (0, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break


if __name__ == "__main__":
    main()
