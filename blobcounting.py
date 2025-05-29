import cv2
import numpy as np
from skimage.feature import blob_log
from math import sqrt
from matplotlib import pyplot as plt

# Global variables
points = []
mask = None
reference_frame = None
threshold_value = 70
blob_min_sigma = 5  # Minimum blob radius
blob_max_sigma = 15  # Maximum blob radius
blob_threshold = 0.2  # Blob detection threshold

def click_event(event, x, y, flags, param):
    global points, mask, reference_frame
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        frame_copy = reference_frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(frame_copy, pt, 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(frame_copy, points[i-1], pt, (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(frame_copy, points[-1], points[0], (0, 255, 0), 2)
            mask = np.zeros_like(frame_copy[:,:,0], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(points)], 255)
            print("Mask created! Press any key to continue...")
        cv2.imshow("Select 4 Corners", frame_copy)

def detect_blobs(difference_image):
    # Convert to grayscale and normalize for blob detection
    gray = cv2.cvtColor(difference_image, cv2.COLOR_BGR2GRAY)
    gray_norm = cv2.normalize(gray, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Detect blobs using Laplacian of Gaussian
    blobs = blob_log(gray_norm, 
                    min_sigma=blob_min_sigma,
                    max_sigma=blob_max_sigma,
                    num_sigma=5,
                    threshold=blob_threshold)
    
    # Scale blob radii
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * sqrt(2)
    return blobs

def frame_difference(current_frame, reference_frame, mask=None):
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_current, gray_reference)
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    if mask is not None:
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    # Create output with changes
    output = np.zeros_like(current_frame)
    output[thresh == 255] = current_frame[thresh == 255]
    
    # Detect blobs in the changed regions
    blobs = detect_blobs(output)
    
    return output, blobs

def main():
    global reference_frame, threshold_value
    
    video_path = "64210013_Video_1A.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)  # Frame to use as reference
    ret, reference_frame = cap.read()
    
    if not ret:
        print("Error reading video")
        return
    
    # Create window for point selection
    cv2.namedWindow("Select 4 Corners")
    cv2.setMouseCallback("Select 4 Corners", click_event)
    
    while True:
        cv2.imshow("Select 4 Corners", reference_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("Select 4 Corners", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    if mask is None or len(points) != 4:
        print("Mask not created properly")
        cap.release()
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("Processing video - detecting blobs in changing regions...")
    print("Controls: q=quit, p=pause, s=save, +/-=adjust threshold")
    
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        
        if ret:
            output, blobs = frame_difference(frame, reference_frame, mask)
            
            # Draw ROI boundary
            cv2.polylines(output, [np.array(points)], True, (0, 255, 0), 1)
            
            # Draw detected blobs
            for blob in blobs:
                y, x, r = blob
                cv2.circle(output, (int(x), int(y)), int(r), (255, 0, 0), 2)
            
            # Display blob count and threshold
            cv2.putText(output, f"Blobs: {len(blobs)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output, f"Threshold: {threshold_value}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Blob Detection in ROI", output)
        
        key = cv2.waitKey(30 if not paused else 1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s') and paused:
            cv2.imwrite("blob_detection.png", output)
            print("Snapshot saved")
        elif key == ord('+'):
            threshold_value = min(255, threshold_value + 5)
            print(f"Threshold: {threshold_value}")
        elif key == ord('-'):
            threshold_value = max(0, threshold_value - 5)
            print(f"Threshold: {threshold_value}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()