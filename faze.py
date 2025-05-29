import cv2
import json

# --- Configuration ---
video_path = "Videi/64210013_Video_5X.mp4"     # Replace with your video file
output_json = "JSON/video_phases_5XY.json"
slowdown_factor = 2.0             # 1.0 = normal speed, 2.0 = half speed, 3.0 = slower, etc.

# --- Video Setup ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
delay = int((1000 / fps) * slowdown_factor)  # delay in milliseconds

phases = []
current_phase = {}

def get_timestamp(frame_num):
    seconds = frame_num / fps
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}"

print("Controls:")
print("  SPACE: Pause/Play")
print("  s: Mark START of phase")
print("  e: Mark END of phase and enter label")
print("  q: Quit and export JSON\n")

paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = get_timestamp(frame_number)

        # Draw info on video
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {timestamp}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Video Phase Marker', frame)

    key = cv2.waitKey(delay if not paused else 0) & 0xFF

    if key == ord(' '):  # Spacebar to toggle pause/play
        paused = not paused

    elif key == ord('s'):  # Mark start
        current_phase["start_frame"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        #current_phase["start_time"] = get_timestamp(current_phase["start_frame"])
        
        #print(f"Start marked at frame {current_phase['start_frame']} ({current_phase['start_time']})")
        print(f"Start marked at frame {current_phase['start_frame']}")


    elif key == ord('e'):  # Mark end and save
        current_phase["end_frame"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        #current_phase["end_time"] = get_timestamp(current_phase["end_frame"])
        label = input("Enter label for this phase: ")
        current_phase["label"] = label
        phases.append(current_phase.copy())
        print(f"Phase saved: {current_phase}")
        current_phase.clear()

    elif key == ord('q'):  # Quit
        print("Exiting and saving JSON...")
        break

cap.release()
cv2.destroyAllWindows()

# --- Export to JSON ---
with open(output_json, "w") as f:
    json.dump(phases, f, indent=4)

print(f"\nFinished playing video.")
print(f"Total number of frames in video: {total_frames}")

print(f"\nExported {len(phases)} phases to {output_json}")
