import cv2
import datetime

# Load video
video_path = "your_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video FPS
fps = cap.get(cv2.CAP_PROP_FPS)

frame_number = 0
blood_timestamps = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color range for blood detection
    lower_red1 = (0, 50, 50)
    upper_red1 = (10, 255, 255)
    lower_red2 = (160, 50, 50)
    upper_red2 = (180, 255, 255)

    # Threshold the image
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Count red pixels
    red_pixels = cv2.countNonZero(mask)
    total_pixels = frame.shape[0] * frame.shape[1]
    red_ratio = red_pixels / total_pixels

    # Threshold for considering as blood
    if red_ratio > 0.02:  # Adjust based on your video
        timestamp_sec = frame_number / fps
        time_str = str(datetime.timedelta(seconds=timestamp_sec))
        blood_timestamps.append(time_str)
        print(f"Blood detected at {time_str}")

    frame_number += 1

cap.release()
