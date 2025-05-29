import cv2
import time

# Initialize two cameras
camera_2 = cv2.VideoCapture(1)  # First camera
camera_1 = cv2.VideoCapture(2)  # Second camera

# Check if cameras opened successfully
if not camera_1.isOpened() or not camera_2.isOpened():
    print("Error: One or both cameras could not be opened.")
    exit()

# Define the codec and create VideoWriter objects to save videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = None
out2 = None

recording = False  # Flag to track recording status

try:
    while True:
        # Read frames from both cameras
        ret1, frame1 = camera_1.read()
        ret2, frame2 = camera_2.read()

        if ret1 and ret2:
            # Show frames in separate windows
            cv2.imshow('Camera 1', frame1)
            cv2.imshow('Camera 2', frame2)

            # If recording, save frames to video files
            if recording:
                out1.write(frame1)
                out2.write(frame2)


        # Check for keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the program
            break
        elif key == ord('r') and not recording:  # Start recording on 'R'
            print("Recording started.")
            recording = True
            start_time = time.time()

            # Initialize VideoWriter objects
            out1 = cv2.VideoWriter('64210013_Video_E2.mp4', fourcc, 20.0, (640, 480)) 
            out2 = cv2.VideoWriter('64210013_Video_F2.mp4', fourcc, 20.0, (640, 480))
            
        elif key == ord('t') and recording:  # Stop recording on 'T'
            print("Recording stopped.")
            recording = False
            out1.release()
            out2.release()
            out1, out2 = None, None  # Reset video writers

except KeyboardInterrupt:
    print("Process interrupted.")
finally:
    # Release resources
    camera_1.release()
    camera_2.release()
    if out1 is not None:
        out1.release()
    if out2 is not None:
        out2.release()
    cv2.destroyAllWindows()