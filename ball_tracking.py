import cv2
import numpy as np


def initial_ball_coordinates(frame):
    
    # Apply any necessary preprocessing steps to enhance the ball's visibility
    
    # Convert the frame to grayscale for easier processing
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to segment the ball from the background
    _, frame = cv2.threshosld(frame, 0, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through the contours and find the largest one (assuming it's the ball)
    max_contour = max(contours, key=cv2.contourArea)
    
    # Find the centroid of the largest contour
    M = cv2.moments(max_contour)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    
    # Return the initial position as a tuple (x, y)
    return (centroid_x, centroid_y)



def track_ball(frame, initial_position=None):
    height, width, _ = frame.shape

    # Apply Gaussian Blur to reduce noise
    blurred_frame = cv2.GaussianBlur(frame, (11, 11), 0)
    rgb_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB

    low = 180  # Adjust if needed
    lower_color = np.array([low, low, low])
    upper_color = np.array([255, 255, 255])

    mask = cv2.inRange(rgb_frame, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 1500:  # Adjust these values based on the expected size of the golf ball
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.7:  # Adjust based on circularity threshold
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if initial_position is None or (abs(initial_position[0] - x) < 50 and abs(initial_position[1] - y) < 50):  # Adjust threshold as necessary
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(frame, center, radius, (0, 255, 255), 2)
                    certainty = int(circularity * 100)
                    text = f"Ball ({certainty}%), center = ({center[0]}, {center[1]})"
                    cv2.putText(frame, text, (center[0] - 50, center[1] - radius - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# def track_ball(frame, initial_position=None):
#     height, width, _ = frame.shape
#     # half_height = (height//6) *4
#     half_height = height 

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB

#     low = 200
#     lower_color = np.array([low, low, low])  # Adjust these values based on your desired color range (in RGB format)
#     upper_color = np.array([255, 255, 255])  # Adjust these values based on your desired color range (in RGB format)

#     mask = cv2.inRange(rgb_frame, lower_color, upper_color)
#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)

#     contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         area = cv2.contourArea(contour)
#         perimeter = cv2.arcLength(contour, True)
#         circularity = 4 * np.pi * (area / (perimeter * perimeter))

#         if circularity > 0.50:  # Adjust this value based on the circularity threshold
#             (x, y), radius = cv2.minEnclosingCircle(contour)
#             center = (int(x), int(y))
#             radius = int(radius) + 10  # Increase the radius to make the circle larger

#             if center[1] > half_height:  # Only consider circles in the bottom half
#                 cv2.circle(frame, center, radius, (0, 255, 255), 2)  # Draw a red circle
#                 certainty = int(circularity * 100)  # Calculate the degree of certainty
#                 text = f"Ball ({certainty}%), center = ({center[0]}, {center[1]})"
#                 cv2.putText(frame, text, (center[0] - 50, center[1] - radius - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     return frame
