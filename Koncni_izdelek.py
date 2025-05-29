from GITHUB.marking import CornerMarker
import cv2
import mediapipe as mp
import time
from collections import deque
import numpy as np
import json
import os
# Configuration parameters
X = 5  # Number of frames to average over
video_path = "Videi/64210013_Video_1A.mp4"
threshold_speed = 70  # Original speed threshold
min_segment_frames = 8  # Minimum frames for a valid segment
json_print = False


# Playback control
playback_speed = 1
frame_delay = int((1.0 / playback_speed) * 33)
effective_threshold = threshold_speed * playback_speed
space_mode = False  # Will become True after space is pressed once

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video.")
    exit()

# Store previous positions and velocities
prev_positions = {5: None, 9: None, 13: None, 17: None}
velocity_history = {5: deque(maxlen=X), 9: deque(maxlen=X), 13: deque(maxlen=X), 17: deque(maxlen=X)}

# Phase 1: Corner marking
ret, first_frame = cap.read()
if not ret:
    print("Failed to read first frame.")
    exit()

marker = CornerMarker(first_frame)
cv2.namedWindow("Click 4 Corners")
cv2.setMouseCallback("Click 4 Corners", marker.click_event)

while True:
    cv2.imshow("Click 4 Corners", marker.img_copy)
    key = cv2.waitKey(1) & 0xFF
    if marker.result_ready and (key == 13 or key == 32):  # ENTER or SPACE
        break
    if key == 27:  # ESC
        cv2.destroyAllWindows()
        cap.release()
        exit()

points, (mid1, mid2) = marker.get_points_and_midline()
print("Corner Points:", points)
print("Midpoints:", mid1, mid2)
cv2.destroyWindow("Click 4 Corners")

# Define interaction areas
grab_square = np.array([points[0], points[1], mid2, mid1])
drop_square = np.array([points[3], points[2], mid2, mid1])

# Show confirmation and reset video
final_img = marker.draw_final_result()
cv2.imshow("Final Rectangle", final_img)
cv2.waitKey(500)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Phase 2: Hand tracking setup
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Tracking variables
pTime = time.time()
current_gesture = "prazna_roka"
new_gesture = "prazna_roka"
gesture_start_frame = 0
gesture_segments = []
frame_count = 0
roka_polna = 0
pins = 0
space_pressed = False

def is_point_inside_polygon(point, polygon):
    if point is None or polygon is None:
        return False
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Main processing loop
while True:
    frame_count += 1
    success, img = cap.read()
    if not success:
        break
    
    # Process frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Draw interaction areas
    cv2.polylines(img, [grab_square], True, (0, 255, 0), 2)
    cv2.polylines(img, [drop_square], True, (0, 0, 255), 2)

    current_positions = {}
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                if id in [5, 9, 13, 17]:
                    current_positions[id] = (cx, cy)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                
                if id in [4, 8, 12, 16, 20]:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # Calculate velocities
            for id in [5, 9, 13, 17]:
                if id in current_positions and prev_positions[id] is not None:
                    prev_x, prev_y = prev_positions[id]
                    curr_x, curr_y = current_positions[id]
                    displacement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    time_diff = (time.time() - pTime) / playback_speed if pTime != 0 else 1/30
                    velocity = displacement / time_diff
                    velocity_history[id].append(velocity)
                    
                    cv2.putText(img, f"{id}:{velocity:.1f}", (curr_x, curr_y), 
                               cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)

            # Update previous positions
            for id in current_positions:
                prev_positions[id] = current_positions[id]

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate moving averages
    moving_averages = {}
    for id in [5, 9, 13, 17]:
        moving_averages[id] = sum(velocity_history[id])/len(velocity_history[id]) if velocity_history[id] else 0
    
    # Display information
    y_offset = 120
    for id in [5, 9, 13, 17]:
        cv2.putText(img, f"{id} (last {X}): {moving_averages[id]:.1f} px/s", 
                   (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        y_offset += 20

    # Update timing
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    # Display overlay
    #cv2.putText(img, f"FPS: {int(fps)}", (10, 20), 
    #           cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
    cv2.putText(img, f"Gesture: {current_gesture}", (10, 40),
               cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
    #cv2.putText(img, f"Speed: {playback_speed:.1f}x", (10, 60),
    #           cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
    #cv2.putText(img, f"Pini: {pins}", (10, 80),
    #           cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

    # Gesture detection
    pos_9 = current_positions.get(9, None)
    pos_13 = current_positions.get(13, None)
    
    in_grab_area = pos_9 and pos_13 and (is_point_inside_polygon(pos_13, grab_square) and is_point_inside_polygon(pos_9, grab_square))
    in_drop_area = pos_9 and pos_13 and (is_point_inside_polygon(pos_13, drop_square) and is_point_inside_polygon(pos_9, drop_square))
    in_grab_area13 = pos_13 and is_point_inside_polygon(pos_13, grab_square)

    avg_velocity = sum(moving_averages.values())/4 if moving_averages else 0

    key = cv2.waitKey(frame_delay) & 0xFF
    if key == 32:  # On first spacebar press, switch mode
        space_pressed = True

    # Gesture state machine
    if not space_pressed: # or pins < 9:
        if avg_velocity > effective_threshold:
            if roka_polna == 1:
                new_gesture = "prenos_pina"
            else:
                new_gesture = "prazna_roka"
        elif in_grab_area:
            new_gesture = "prijemanje_pina"
            roka_polna = 1
        elif in_drop_area:
            new_gesture = "odlaganje_pina"
            roka_polna = 0
    if space_pressed:
        if in_drop_area: # and avg_velocity < 1.6 * effective_threshold:
            roka_polna = 0
            #new_gesture = "prazna_roka"
        elif in_grab_area: #and avg_velocity < 1.6 * effective_threshold:
            roka_polna = 1
            #new_gesture = "prenos_pina"
        if roka_polna == 1:
            new_gesture = "prenos_pina"
        elif roka_polna == 0:
            new_gesture = "prazna_roka"
    # Handle gesture transitions
    if new_gesture != current_gesture:
        if (frame_count - gesture_start_frame) >= min_segment_frames:
            gesture_segments.append({
                "start_frame": gesture_start_frame,
                "end_frame": frame_count - 1,
                "label": current_gesture
            })
            #if new_gesture == "odlaganje_pina":
            #    pins += 1
            current_gesture = new_gesture
            gesture_start_frame = frame_count
        elif frame_count == gesture_start_frame:
            current_gesture = new_gesture

    # Display and handle controls
    cv2.imshow("Hand Tracking", img)
    if key == ord('q'):
        break
    elif key == ord('+'):  # Speed up
        playback_speed = min(2.0, playback_speed + 0.1)
        frame_delay = int((1.0 / playback_speed) * 33)
        effective_threshold = threshold_speed * playback_speed
    elif key == ord('-'):  # Slow down
        playback_speed = max(0.1, playback_speed - 0.1)
        frame_delay = int((1.0 / playback_speed) * 33)
        effective_threshold = threshold_speed * playback_speed

# Final processing
if (frame_count - gesture_start_frame) >= min_segment_frames:
    gesture_segments.append({
        "start_frame": gesture_start_frame,
        "end_frame": frame_count,
        "label": current_gesture
    })

# Extract the video filename without extension
video_filename = os.path.splitext(os.path.basename(video_path))[0]

# Create the output JSON filename
if json_print == True:
    output_json = f'JSON/{video_filename}_prediction.json'

    # Save the gesture segments
    with open(output_json, 'w') as f:
        json.dump(gesture_segments, f, indent=4)

    print(f"Saved gesture segments to {output_json}")
cap.release()
cv2.destroyAllWindows()