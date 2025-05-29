import cv2
import numpy as np

# Open the video
cap = cv2.VideoCapture("64210013_Video_1A.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_number = 4  # Change this to the frame number you want

# Set the video capture to the specific frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Read the specified frame
success, img = cap.read()

# Check if the frame was successfully read
if not success:
    print(f"Error: Could not read frame {frame_number}.")
    cap.release()
    exit()

# Get the image dimensions
height, width = img.shape[:2]

# Define the central region (50% of the frame size)
roi_height = height // 2
roi_width = width // 2
top_left_x = (width - roi_width) // 2
top_left_y = (height - roi_height) // 2

# Crop the image to the central region
central_img = img[top_left_y:top_left_y + roi_height, top_left_x:top_left_x + roi_width]

# Step 1: Convert to grayscale
gray = cv2.cvtColor(central_img, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Perform Canny edge detection
edges = cv2.Canny(blur, threshold1=100, threshold2=190)

# Step 4: Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: Sort contours by area and keep the largest one (assuming it's the box)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Step 6: Approximate the contour to a polygon (to detect a rectangular shape)
for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the approximated contour has 4 corners (rectangle)
    if len(approx) == 4:
        # Draw the detected box's border on the original image (not the cropped one)
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
        break  # Assuming there's only one box, break the loop

# Display the result
cv2.imshow("Detected Box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release the video capture object
cap.release()

print(hierarchy)
