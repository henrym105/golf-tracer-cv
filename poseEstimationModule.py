import cv2
import mediapipe as mp
import time
import os

class poseDetctor:
    def __init__(self, mode = False, upBody = False, smooth = True, detectionConf = 0.5, trackingConf = 0.5 ) -> None:
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, 
                                     min_detection_confidence=self.detectionConf, 
                                     min_tracking_confidence=self.trackingConf)

    def findPose(self, img, draw = True):
        """Find the pose, set draw = False if do not want to draw on the image"""

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        
        if draw and results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

def main():
    cap = cv2.VideoCapture(os.path.join('Videos', 'processed', 'putts1.mp4'))
    pTime = 0

    detector = poseDetctor()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        
        cv2.imshow("Image", img)
        # cv2.putText(img, text=str(int(fps)), org = (70, 50), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 3, color = (255, 0, 0), thickness = 3)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()