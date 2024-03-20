import cv2
import os
import time
from preprocessing import preprocess_frame
from ball_tracking import track_ball
import poseEstimationModule as pem

def main():
    video_name = 'tw.mp4'
    video_path = os.path.join('Videos', 'raw', video_name)
    cap = cv2.VideoCapture(video_path)
    detector = pem.poseDetctor()
    pTime = 0

    while cap.isOpened():
        success, frame = cap.read()

        if not success: 
            break

        frame = preprocess_frame(frame)
        frame = track_ball(frame)
        frame = detector.findPose(frame)

        # calculate fps and overlay on video
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(frame, 
                    text = f"{str(int(fps))} fps", 
                    org = (10, 20), 
                    fontFace = cv2.FONT_HERSHEY_PLAIN, 
                    fontScale = 1, 
                    color = (255, 255, 255), 
                    thickness = 1 )

        # show the modified video frame
        cv2.imshow('Ball Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

