import cv2

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (800, 600))
    # Apply additional preprocessing steps if needed
    return resized_frame
