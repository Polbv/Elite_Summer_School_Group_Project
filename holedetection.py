import cv2
import numpy as np
import time
import random
import json
# Start timer
start_time = time.time()

# Load the image
image_path = "media/raw/hole2.jpeg"  # Change to your image file path
image = cv2.imread(image_path)
N=20
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
mask = np.zeros(image.shape[:2], dtype=np.uint8)
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reduce noise to avoid false circle detection
gray_blurred = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(
    gray_blurred,
    cv2.HOUGH_GRADIENT,
    dp=0.64,         # Inverse ratio of the accumulator resolution
    minDist=17,     # Minimum distance between detected centers
    param1=59,           # Upper threshold for Canny edge detection
    param2=35,      # Threshold for center detection
    minRadius=60,    # Minimum circle radius
    maxRadius=100    # Maximum circle radius
)

# If some circles are detected, draw them
circle_id=0
circle_mean=[]
if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
       
        # Draw the circle
        
        cv2.circle(mask, (x, y), r, 255, -1)
        # Draw the center
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
        mean_val = cv2.mean(gray_blurred, mask=mask) 
        print(f"circle id: { circle_id}, mean value: {mean_val}")
        

# End timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")


max_width = 800
max_height = 800
height, width = image.shape[:2]
scaling_factor = min(max_width / width, max_height / height, 1)
if scaling_factor < 1:
    display_image = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))
else:
    display_image = image

cv2.imshow("Detected Circles", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# #save image
date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
# cv2.imwrite(f"/media/processed/{date_time}.jpg",display_image)
